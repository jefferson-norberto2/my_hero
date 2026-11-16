import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_context_calculator.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_memory_parser.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_thinking_parser.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Exception thrown when the Ollama server explicitly rejects a `/api/pull` request.
class OllamaPullServerException implements Exception {
  final String message;

  const OllamaPullServerException(this.message);

  @override
  String toString() => message;
}

/// Remote datasource communicating with the Ollama HTTP API.
abstract class OllamaRemoteDatasource {
  String get baseUrl;
  void updateBaseUrl(String url);
  Future<bool> checkHealth([String? customUrl]);
  Future<List<OllamaModel>> getAvailableModels();
  Future<List<String>> getLoadedModelNames();
  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName);
  Future<int> getModelMaxContextLength(String modelName);
  Future<void> preloadModel(String modelName);
  Stream<ChatStreamChunk> generateChatResponseAsync({
    required String modelName,
    required List<Map<String, dynamic>> messages,
    int? numCtx,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  });
  Stream<ModelDownloadState> pullModel(String modelName);
  void stopGeneration();
  void cancelModelPull();
  void close();
}

class OllamaRemoteDatasourceImpl implements OllamaRemoteDatasource {
  final DioClient _dioClient;
  CancelToken? _chatCancelToken;
  CancelToken? _pullCancelToken;
  String _baseUrl;

  OllamaRemoteDatasourceImpl(this._dioClient, {String? baseUrl})
      : _baseUrl = baseUrl ?? AppConstants.ollamaBaseUrl;

  @override
  String get baseUrl => _baseUrl;

  @override
  void updateBaseUrl(String url) {
    _baseUrl = url;
  }

  @override
  Future<bool> checkHealth([String? customUrl]) async {
    final targetUrl = (customUrl != null && customUrl.trim().isNotEmpty)
        ? customUrl.trim()
        : _baseUrl;
    try {
      final response = await _dioClient.dio.get(
        '$targetUrl/',
        options: Options(
          sendTimeout: const Duration(seconds: 3),
          receiveTimeout: const Duration(seconds: 3),
        ),
      );
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<Map<String, dynamic>>> _getPsModels() async {
    try {
      final response = await _dioClient.dio.get(
        '$_baseUrl/api/ps',
        options: Options(
          sendTimeout: const Duration(seconds: 5),
          receiveTimeout: const Duration(seconds: 5),
        ),
      );
      if (response.statusCode == 200 && response.data is Map<String, dynamic>) {
        final models = (response.data['models'] as List<dynamic>?) ?? [];
        return models.cast<Map<String, dynamic>>();
      }
      return [];
    } catch (_) {
      return [];
    }
  }

  @override
  Future<List<String>> getLoadedModelNames() async {
    final models = await _getPsModels();
    return models
        .map((m) => (m['name'] as String? ?? m['model'] as String? ?? ''))
        .where((s) => s.isNotEmpty)
        .toList();
  }

  @override
  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName) async {
    final models = await _getPsModels();
    for (final m in models) {
      final name = m['name'] as String? ?? '';
      final model = m['model'] as String? ?? '';
      if (name == modelName ||
          model == modelName ||
          name.startsWith('$modelName:') ||
          modelName.startsWith('$name:')) {
        return m;
      }
    }
    return null;
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() async {
    final psModels = await _getPsModels();
    final response = await _dioClient.dio.get(
      '$_baseUrl/api/tags',
      options: Options(
        sendTimeout: const Duration(seconds: 5),
        receiveTimeout: const Duration(seconds: 5),
      ),
    );

    if (response.statusCode == 200 && response.data is Map<String, dynamic>) {
      final modelsJson = (response.data['models'] as List<dynamic>?) ?? [];
      return modelsJson.map((m) {
        final map = m as Map<String, dynamic>;
        final name = map['name'] as String? ?? '';
        final model = map['model'] as String? ?? '';

        final loadedMatch = psModels.where(
          (loaded) {
            final lName = loaded['name'] as String? ?? '';
            final lModel = loaded['model'] as String? ?? '';
            return lName == name ||
                lModel == model ||
                lName.startsWith('$name:') ||
                name.startsWith('$lName:');
          },
        ).firstOrNull;

        final isLoaded = loadedMatch != null;
        final memoryAllocation = OllamaMemoryParser.parsePsEntry(loadedMatch);
        final contextLength = (loadedMatch?['context_length'] as num?)?.toInt();

        return OllamaModel.fromJson(
          map,
          isLoaded: isLoaded,
          sizeVram: memoryAllocation.vramBytes,
          memoryAllocation: memoryAllocation,
          contextLength: contextLength,
        );
      }).toList();
    }
    return [];
  }

  @override
  Future<int> getModelMaxContextLength(String modelName) async {
    if (modelName.trim().isEmpty) {
      return ContextSettings.fallbackMaxContext;
    }

    try {
      final response = await _dioClient.dio.post(
        '$_baseUrl/api/show',
        data: {'name': modelName},
        options: Options(
          sendTimeout: const Duration(seconds: 5),
          receiveTimeout: const Duration(seconds: 5),
        ),
      );
      if (response.statusCode == 200 && response.data is Map<String, dynamic>) {
        return OllamaContextCalculator.parseMaxContextFromShowResponse(
          response.data as Map<String, dynamic>,
        );
      }
      return ContextSettings.fallbackMaxContext;
    } catch (_) {
      return ContextSettings.fallbackMaxContext;
    }
  }

  @override
  Future<void> preloadModel(String modelName) async {
    // Sending an empty generate request instructs Ollama to load model weights into memory/VRAM.
    await _dioClient.dio.post(
      '$_baseUrl/api/generate',
      data: {'model': modelName},
      options: Options(
        responseType: ResponseType.json,
        sendTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(minutes: 5),
      ),
    );
  }

  @override
  Stream<ChatStreamChunk> generateChatResponseAsync({
    required String modelName,
    required List<Map<String, dynamic>> messages,
    int? numCtx,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  }) async* {
    _chatCancelToken?.cancel();
    final cancelToken = CancelToken();
    _chatCancelToken = cancelToken;

    final response = await _dioClient.dio.post<ResponseBody>(
      '$_baseUrl/api/chat',
      data: {
        'model': modelName,
        'messages': messages,
        'stream': true,
        if (numCtx != null && numCtx > 0) 'options': {'num_ctx': numCtx},
      },
      options: Options(
        responseType: ResponseType.stream,
        sendTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(minutes: 10),
      ),
      cancelToken: cancelToken,
    );

    final stream = response.data?.stream;
    if (stream == null) return;

    final parser = OllamaThinkingParser();
    var sawDoneChunk = false;

    final lineStream = stream
        .cast<List<int>>()
        .transform(utf8.decoder)
        .transform(const LineSplitter());

    await for (final line in lineStream) {
      if (cancelToken.isCancelled) break;
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      try {
        final json = jsonDecode(trimmed) as Map<String, dynamic>;

        if (json['done'] == true) {
          sawDoneChunk = true;
          final promptCount = (json['prompt_eval_count'] as num?)?.toInt() ?? 0;
          final evalCount = (json['eval_count'] as num?)?.toInt() ?? 0;
          if (promptCount > 0 || evalCount > 0) {
            onMetrics?.call(promptCount, evalCount);
          }
        }

        final chunk = parser.parseJsonChunk(json);
        if (!chunk.isEmpty) {
          yield chunk;
        }
      } catch (_) {
        // Ignore malformed NDJSON chunks.
      }
    }

    if (!cancelToken.isCancelled && !sawDoneChunk) {
      final tail = parser.flush();
      if (!tail.isEmpty) {
        yield tail;
      }
    }
  }

  @override
  Stream<ModelDownloadState> pullModel(String modelName) async* {
    _pullCancelToken?.cancel('Replaced by new model pull');
    final cancelToken = CancelToken();
    _pullCancelToken = cancelToken;

    final response = await _dioClient.dio.post<ResponseBody>(
      '$_baseUrl/api/pull',
      data: {
        'model': modelName,
        'stream': true,
      },
      options: Options(
        responseType: ResponseType.stream,
        sendTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(minutes: 30),
      ),
      cancelToken: cancelToken,
    );

    final stream = response.data?.stream;
    if (stream == null) {
      throw const HttpException('Empty response stream from /api/pull.');
    }

    var receivedSuccess = false;
    final lineStream = stream
        .cast<List<int>>()
        .transform(utf8.decoder)
        .transform(const LineSplitter());

    await for (final line in lineStream) {
      if (cancelToken.isCancelled) break;
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      final Map<String, dynamic> json;
      try {
        final decoded = jsonDecode(trimmed);
        if (decoded is! Map<String, dynamic>) continue;
        json = decoded;
      } catch (_) {
        // Ignore malformed NDJSON chunks.
        continue;
      }

      final error = json['error'] as String?;
      if (error != null && error.trim().isNotEmpty) {
        throw OllamaPullServerException(error.trim());
      }

      final status = (json['status'] as String? ?? '').trim();
      final total = (json['total'] as num?)?.toInt() ?? 0;
      final completed = (json['completed'] as num?)?.toInt() ?? 0;

      if (status == 'success') {
        receivedSuccess = true;
        yield ModelDownloadState(
          modelName: modelName,
          status: ModelDownloadStatus.completed,
          statusMessage: 'Modelo "$modelName" baixado com sucesso!',
          completedBytes: total > 0 ? total : completed,
          totalBytes: total,
        );
      } else {
        yield ModelDownloadState(
          modelName: modelName,
          status: ModelDownloadStatus.downloading,
          statusMessage: status.isNotEmpty ? status : 'Baixando modelo...',
          completedBytes: completed,
          totalBytes: total,
        );
      }
    }

    if (cancelToken.isCancelled) {
      return;
    }

    if (!receivedSuccess) {
      throw const HttpException(
        'Connection closed before model download completed.',
      );
    }
  }

  @override
  void stopGeneration() {
    _chatCancelToken?.cancel('Cancelled by user');
    _chatCancelToken = null;
  }

  @override
  void cancelModelPull() {
    _pullCancelToken?.cancel('Cancelled by user');
    _pullCancelToken = null;
  }

  @override
  void close() {
    stopGeneration();
    cancelModelPull();
  }
}
