import 'dart:async';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/services/foreground_task_manager.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';

/// Coordinates Ollama model downloads (`/api/pull`) in the data layer with
/// automatic reconnection on dropped connections and Android foreground support.
class ModelDownloadService {
  /// Maximum automatic reconnection attempts after the initial request drops.
  static const int maxReconnectAttempts = 3;

  /// Default wait duration before each reconnection attempt.
  static const Duration defaultRetryDelay = Duration(seconds: 2);

  final OllamaRemoteDatasource _ollamaDatasource;
  final ForegroundTaskManager _foregroundTaskManager;
  final Duration _retryDelay;

  final StreamController<ModelDownloadState> _stateController =
      StreamController<ModelDownloadState>.broadcast();

  ModelDownloadState _currentState = const ModelDownloadState.idle();
  bool _isCancelRequested = false;
  Completer<ModelDownloadState>? _activeDownloadCompleter;
  Completer<void>? _retryDelayCompleter;

  ModelDownloadService({
    required OllamaRemoteDatasource ollamaDatasource,
    ForegroundTaskManager? foregroundTaskManager,
    Duration retryDelay = defaultRetryDelay,
  })  : _ollamaDatasource = ollamaDatasource,
        _foregroundTaskManager =
            foregroundTaskManager ?? ForegroundTaskManager(),
        _retryDelay = retryDelay;

  /// Latest emitted download state snapshot.
  ModelDownloadState get currentState => _currentState;

  /// Broadcast stream of download state updates for UI observers.
  Stream<ModelDownloadState> get stateStream => _stateController.stream;

  /// Starts downloading [modelName] via `POST /api/pull`, retrying up to
  /// [maxReconnectAttempts] times if the HTTP stream drops unexpectedly.
  Future<ModelDownloadState> downloadModel(String modelName) {
    final trimmedName = modelName.trim();
    if (trimmedName.isEmpty) {
      throw ArgumentError.value(
        modelName,
        'modelName',
        'Model name cannot be empty.',
      );
    }

    final existingCompleter = _activeDownloadCompleter;
    if (existingCompleter != null && !existingCompleter.isCompleted) {
      return existingCompleter.future;
    }

    final completer = Completer<ModelDownloadState>();
    _activeDownloadCompleter = completer;
    _isCancelRequested = false;

    unawaited(_executeDownload(trimmedName, completer));
    return completer.future;
  }

  /// Cancels any active model download or pending reconnection wait.
  Future<void> cancelDownload() async {
    _isCancelRequested = true;
    _ollamaDatasource.cancelModelPull();

    final delayCompleter = _retryDelayCompleter;
    if (delayCompleter != null && !delayCompleter.isCompleted) {
      delayCompleter.complete();
    }

    final activeCompleter = _activeDownloadCompleter;
    if (activeCompleter != null && !activeCompleter.isCompleted) {
      await activeCompleter.future;
    }
  }

  /// Releases internal broadcast stream resources.
  void dispose() {
    if (!_stateController.isClosed) {
      _stateController.close();
    }
  }

  Future<void> _executeDownload(
    String modelName,
    Completer<ModelDownloadState> completer,
  ) async {
    var lastTotalBytes = 0;
    var lastCompletedBytes = 0;

    _emitState(
      ModelDownloadState(
        modelName: modelName,
        status: ModelDownloadStatus.downloading,
        statusMessage: 'Iniciando download de $modelName...',
        maxRetries: maxReconnectAttempts,
      ),
    );

    await _foregroundTaskManager.beginModelDownload(modelName);

    try {
      for (var attempt = 0; attempt <= maxReconnectAttempts; attempt++) {
        if (_isCancelRequested) {
          _emitCancelled(modelName, lastCompletedBytes, lastTotalBytes);
          return;
        }

        if (attempt > 0) {
          _emitState(
            ModelDownloadState(
              modelName: modelName,
              status: ModelDownloadStatus.reconnecting,
              statusMessage:
                  'Conexão perdida. Reconectando ($attempt/$maxReconnectAttempts)...',
              completedBytes: lastCompletedBytes,
              totalBytes: lastTotalBytes,
              retryAttempt: attempt,
              maxRetries: maxReconnectAttempts,
            ),
          );
          await _foregroundTaskManager.updateModelDownloadProgress(
            modelName,
            statusText:
                'Reconectando download de $modelName ($attempt/$maxReconnectAttempts)...',
          );

          await _waitRetryDelay();
          if (_isCancelRequested) {
            _emitCancelled(modelName, lastCompletedBytes, lastTotalBytes);
            return;
          }
        }

        try {
          await for (final chunk in _ollamaDatasource.pullModel(modelName)) {
            if (_isCancelRequested) {
              _emitCancelled(modelName, lastCompletedBytes, lastTotalBytes);
              return;
            }

            if (chunk.totalBytes > 0) {
              lastTotalBytes = chunk.totalBytes;
              lastCompletedBytes = chunk.completedBytes;
            }

            final effectiveTotal =
                chunk.totalBytes > 0 ? chunk.totalBytes : lastTotalBytes;
            final effectiveCompleted =
                chunk.totalBytes > 0 ? chunk.completedBytes : lastCompletedBytes;

            final nextState = chunk.copyWith(
              completedBytes: effectiveCompleted,
              totalBytes: effectiveTotal,
              retryAttempt: attempt,
              maxRetries: maxReconnectAttempts,
            );
            _emitState(nextState);

            if (nextState.status == ModelDownloadStatus.completed) {
              return;
            }

            await _foregroundTaskManager.updateModelDownloadProgress(
              modelName,
              percent: nextState.progressPercent,
            );
          }

          if (_isCancelRequested) {
            _emitCancelled(modelName, lastCompletedBytes, lastTotalBytes);
            return;
          }

          if (_currentState.status == ModelDownloadStatus.completed) {
            return;
          }

          // Stream terminated without receiving a completion chunk; retry if possible.
          throw const HttpException(
            'Download stream closed before completion.',
          );
        } catch (error) {
          if (_isCancelRequested || _isCancellationError(error)) {
            _emitCancelled(modelName, lastCompletedBytes, lastTotalBytes);
            return;
          }

          final isPermanentFailure = error is OllamaPullServerException ||
              _isNonRetryableClientError(error);
          final hasRetriesLeft = attempt < maxReconnectAttempts;

          if (isPermanentFailure || !hasRetriesLeft) {
            _emitState(
              ModelDownloadState(
                modelName: modelName,
                status: ModelDownloadStatus.error,
                statusMessage: 'Falha ao baixar $modelName',
                completedBytes: lastCompletedBytes,
                totalBytes: lastTotalBytes,
                retryAttempt: attempt,
                maxRetries: maxReconnectAttempts,
                errorMessage: _formatErrorMessage(error),
              ),
            );
            return;
          }
        }
      }
    } finally {
      await _foregroundTaskManager.endModelDownload();
      if (!completer.isCompleted) {
        completer.complete(_currentState);
      }
      if (identical(_activeDownloadCompleter, completer)) {
        _activeDownloadCompleter = null;
      }
    }
  }

  Future<void> _waitRetryDelay() async {
    if (_retryDelay <= Duration.zero) return;
    final delayCompleter = Completer<void>();
    _retryDelayCompleter = delayCompleter;
    final timer = Timer(_retryDelay, () {
      if (!delayCompleter.isCompleted) {
        delayCompleter.complete();
      }
    });
    try {
      await delayCompleter.future;
    } finally {
      timer.cancel();
      if (identical(_retryDelayCompleter, delayCompleter)) {
        _retryDelayCompleter = null;
      }
    }
  }

  void _emitCancelled(String modelName, int completedBytes, int totalBytes) {
    _emitState(
      ModelDownloadState(
        modelName: modelName,
        status: ModelDownloadStatus.cancelled,
        statusMessage: 'Download de $modelName cancelado.',
        completedBytes: completedBytes,
        totalBytes: totalBytes,
        maxRetries: maxReconnectAttempts,
      ),
    );
  }

  void _emitState(ModelDownloadState state) {
    _currentState = state;
    if (!_stateController.isClosed) {
      _stateController.add(state);
    }
  }

  bool _isCancellationError(Object error) {
    return error is DioException && error.type == DioExceptionType.cancel;
  }

  bool _isNonRetryableClientError(Object error) {
    if (error is DioException && error.type == DioExceptionType.badResponse) {
      final statusCode = error.response?.statusCode ?? 0;
      return statusCode >= 400 && statusCode < 500 && statusCode != 408;
    }
    return false;
  }

  String _formatErrorMessage(Object error) {
    if (error is OllamaPullServerException) {
      return error.message;
    }
    if (error is DioException) {
      return error.message ?? error.toString();
    }
    return error.toString();
  }
}
