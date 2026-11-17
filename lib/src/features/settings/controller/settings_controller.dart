import 'dart:async';
import 'dart:math' as math;
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/core/utils/url_utils.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Manages reactive draft state and actions for the settings screen.
class SettingsController {
  final SettingsRepository _repository;

  SettingsController({
    required SettingsRepository repository,
  }) : _repository = repository;

  final ollamaBaseUrl = signal<String>(AppConstants.ollamaBaseUrl);
  final availableModels = signal<List<OllamaModel>>([]);
  final selectedModelName = signal<String>('');
  final contextMode = signal<ContextMode>(ContextMode.automatic);
  final manualContextLength =
      signal<int>(ContextSettings.fallbackMaxContext);
  final isOllamaConnected = signal<bool>(true);
  final isModelLoaded = signal<bool>(false);

  final isRefreshingModels = signal<bool>(false);
  final isLoadingModel = signal<bool>(false);
  final loadingProgress = signal<double>(0.0);
  final loadingStatus = signal<String>('');
  final loadErrorMessage = signal<String?>(null);
  final downloadState =
      signal<ModelDownloadState>(const ModelDownloadState.idle());

  final isEditingServer = signal<bool>(false);
  final isTestingConnection = signal<bool>(false);
  final isApplyingServer = signal<bool>(false);
  final connectionTestSuccess = signal<bool?>(null);
  final connectionTestMessage = signal<String?>(null);

  String _initialServerUrl = AppConstants.ollamaBaseUrl;
  StreamSubscription<ModelDownloadState>? _downloadSubscription;
  bool _isDisposed = false;
  bool _didHandleDownloadCompletion = false;

  /// Native maximum context supported by the currently selected model.
  int get activeModelMaxContext {
    final match = availableModels.value
        .where((m) => m.name == selectedModelName.value)
        .firstOrNull;
    if (match == null) return ContextSettings.fallbackMaxContext;
    return math.max(match.maxContextLength, match.contextLength);
  }

  /// Initializes the controller with optional route arguments or loads from storage.
  Future<void> init([SettingsArgs? args]) async {
    _bindDownloadState();

    if (args != null) {
      final initialUrl = args.ollamaBaseUrl.isNotEmpty
          ? args.ollamaBaseUrl
          : _repository.ollamaBaseUrl;
      _initialServerUrl = initialUrl;
      ollamaBaseUrl.value = initialUrl;
      availableModels.value = List<OllamaModel>.from(args.availableModels);
      selectedModelName.value = args.selectedModelName;
      if (selectedModelName.value.isEmpty && availableModels.value.isNotEmpty) {
        selectedModelName.value = availableModels.value.first.name;
      }
      contextMode.value = args.contextMode;
      manualContextLength.value = args.manualContextLength;
      isOllamaConnected.value = args.isOllamaConnected;
      isModelLoaded.value = args.isModelLoaded ||
          availableModels.value.any(
            (m) => m.name == selectedModelName.value && m.isLoaded,
          );
      if (isOllamaConnected.value &&
          selectedModelName.value.isNotEmpty &&
          activeModelMaxContext <= ContextSettings.fallbackMaxContext) {
        await _fetchActiveModelMaxContext(selectedModelName.value);
      } else {
        _clampManualContextLength();
      }
      return;
    }

    _initialServerUrl = _repository.ollamaBaseUrl;
    ollamaBaseUrl.value = _initialServerUrl;
    await loadInitialData();
  }

  /// Loads persisted settings and fetches available models.
  Future<void> loadInitialData() async {
    try {
      final savedUrl = await _repository.loadOllamaBaseUrl();
      _initialServerUrl = savedUrl;
      ollamaBaseUrl.value = savedUrl;
      contextMode.value = await _repository.loadContextMode();
      manualContextLength.value = await _repository.loadManualContextLength();
    } catch (_) {
      // Keep defaults if persisted configuration cannot be read.
    }
    await refreshModels();
  }

  Future<void> selectModel(String modelName) async {
    selectedModelName.value = modelName;
    loadErrorMessage.value = null;
    isModelLoaded.value = availableModels.value.any(
      (m) => m.name == modelName && m.isLoaded,
    );
    if (isOllamaConnected.value && modelName.isNotEmpty) {
      await _fetchActiveModelMaxContext(modelName);
    } else {
      _clampManualContextLength();
    }
  }

  void selectContextMode(ContextMode mode) {
    contextMode.value = mode;
    _clampManualContextLength();
  }

  void selectManualContextLength(int length) {
    contextMode.value = ContextMode.manual;
    manualContextLength.value = length;
    _clampManualContextLength();
  }

  void _clampManualContextLength() {
    final allowed = buildContextPresets(activeModelMaxContext);
    manualContextLength.value = resolveClosestContextPreset(
      manualContextLength.value,
      allowed,
    );
  }

  Future<void> _fetchActiveModelMaxContext(String modelName) async {
    try {
      final fetchedMax = await _repository.getModelMaxContextLength(modelName);
      if (fetchedMax <= 0) return;

      availableModels.value = availableModels.value.map((m) {
        if (m.name == modelName) {
          return m.copyWith(maxContextLength: fetchedMax);
        }
        return m;
      }).toList();
    } catch (_) {
      // Preserve existing fallback capacity when /api/show is unavailable.
    } finally {
      _clampManualContextLength();
    }
  }

  Future<void> refreshModels() async {
    if (isRefreshingModels.value) return;

    isRefreshingModels.value = true;
    loadErrorMessage.value = null;

    try {
      final isHealthy = await _repository.checkOllamaHealth();
      isOllamaConnected.value = isHealthy;

      if (!isHealthy) {
        availableModels.value = [];
        selectedModelName.value = '';
        isModelLoaded.value = false;
        return;
      }

      final models = await _repository.getAvailableModels();
      availableModels.value = models;

      if (models.isNotEmpty) {
        final hasCurrent =
            models.any((m) => m.name == selectedModelName.value);
        if (!hasCurrent) {
          final loadedModel = models.where((m) => m.isLoaded).firstOrNull;
          selectedModelName.value = loadedModel?.name ?? models.first.name;
        }
        isModelLoaded.value = models.any(
          (m) => m.name == selectedModelName.value && m.isLoaded,
        );
        if (selectedModelName.value.isNotEmpty) {
          await _fetchActiveModelMaxContext(selectedModelName.value);
        }
      } else {
        selectedModelName.value = '';
        isModelLoaded.value = false;
      }
    } catch (e) {
      isOllamaConnected.value = false;
      availableModels.value = [];
      isModelLoaded.value = false;
      loadErrorMessage.value = 'Erro ao consultar modelos do Ollama: $e';
    } finally {
      isRefreshingModels.value = false;
    }
  }

  /// Preloads the currently selected model into Ollama memory.
  Future<void> loadModelIntoMemory() async {
    final target = selectedModelName.value.isNotEmpty
        ? selectedModelName.value
        : (availableModels.value.firstOrNull?.name ?? '');

    if (target.isEmpty || isLoadingModel.value) return;

    isLoadingModel.value = true;
    loadingProgress.value = 0.0;
    loadingStatus.value = 'Solicitando carregamento de $target no Ollama...';
    loadErrorMessage.value = null;

    try {
      await _repository.loadModel(
        modelName: target,
        onProgress: (progress, status) {
          loadingProgress.value = progress;
          loadingStatus.value = status;
        },
      );

      selectedModelName.value = target;
      await refreshModels();
      await _fetchActiveModelMaxContext(target);
      isModelLoaded.value = true;

      // Ensure the loaded flag is reflected even if /api/ps has not updated yet.
      final updatedModels = availableModels.value.map((m) {
        if (m.name == target && !m.isLoaded) {
          return m.copyWith(isLoaded: true);
        }
        return m;
      }).toList();
      availableModels.value = updatedModels;
    } catch (e) {
      loadErrorMessage.value = 'Falha ao carregar o modelo no Ollama: $e';
    } finally {
      isLoadingModel.value = false;
    }
  }

  /// Starts downloading [rawModelName] via `/api/pull` in the data layer.
  Future<void> downloadModel(String rawModelName) async {
    final trimmed = rawModelName.trim();
    if (trimmed.isEmpty || downloadState.value.isActive) return;

    _didHandleDownloadCompletion = false;
    loadErrorMessage.value = null;

    try {
      final result = await _repository.downloadModel(trimmed);
      if (_isDisposed) return;
      await _handleDownloadStateUpdate(result);
    } catch (e) {
      if (!_isDisposed) {
        loadErrorMessage.value = 'Falha ao baixar o modelo: $e';
      }
    }
  }

  /// Cancels any active model download running in the data layer.
  Future<void> cancelModelDownload() {
    return _repository.cancelModelDownload();
  }

  void _bindDownloadState() {
    downloadState.value = _repository.currentDownloadState;
    _didHandleDownloadCompletion =
        downloadState.value.status == ModelDownloadStatus.completed;
    _downloadSubscription?.cancel();
    _downloadSubscription = _repository.downloadStateStream.listen((state) {
      if (_isDisposed) return;
      unawaited(_handleDownloadStateUpdate(state));
    });
  }

  Future<void> _handleDownloadStateUpdate(ModelDownloadState state) async {
    if (_isDisposed) return;
    downloadState.value = state;

    if (state.status.isActive) {
      _didHandleDownloadCompletion = false;
    }

    if (state.status == ModelDownloadStatus.completed &&
        !_didHandleDownloadCompletion) {
      _didHandleDownloadCompletion = true;
      loadErrorMessage.value = null;
      if (selectedModelName.value.isEmpty) {
        selectedModelName.value = state.modelName;
      }
      await refreshModels();
    } else if (state.status == ModelDownloadStatus.error &&
        state.errorMessage != null) {
      loadErrorMessage.value =
          'Falha ao baixar o modelo: ${state.errorMessage}';
    }
  }

  /// Detaches UI subscriptions without cancelling active background downloads.
  void dispose() {
    _isDisposed = true;
    _downloadSubscription?.cancel();
    _downloadSubscription = null;
  }

  void toggleServerEditor() {
    isEditingServer.value = !isEditingServer.value;
    clearConnectionTest();
  }

  void clearConnectionTest() {
    if (connectionTestSuccess.value != null ||
        connectionTestMessage.value != null) {
      connectionTestSuccess.value = null;
      connectionTestMessage.value = null;
    }
  }

  Future<bool> testConnection(String rawHost) async {
    final input = rawHost.trim();
    if (input.isEmpty) return false;

    isTestingConnection.value = true;
    connectionTestSuccess.value = null;
    connectionTestMessage.value = null;

    try {
      final normalized = normalizeOllamaUrl(input);
      final isHealthy = await _repository.checkOllamaHealth(normalized);
      connectionTestSuccess.value = isHealthy;
      connectionTestMessage.value = isHealthy
          ? 'Conectado com sucesso em $normalized!'
          : 'Não foi possível conectar em $normalized. Verifique o IP, porta e firewall.';
      return isHealthy;
    } catch (e) {
      connectionTestSuccess.value = false;
      connectionTestMessage.value = 'Erro de conexão: $e';
      return false;
    } finally {
      isTestingConnection.value = false;
    }
  }

  /// Applies a draft server URL without persisting it until the user saves.
  Future<bool> applyServerHost(String rawHost) async {
    isApplyingServer.value = true;
    connectionTestSuccess.value = null;
    connectionTestMessage.value = null;

    try {
      final normalized = normalizeOllamaUrl(rawHost);
      ollamaBaseUrl.value = normalized;
      _repository.updateOllamaBaseUrl(normalized);
      await refreshModels();

      if (isOllamaConnected.value) {
        isEditingServer.value = false;
      } else {
        connectionTestSuccess.value = false;
        connectionTestMessage.value =
            'Não foi possível conectar em $normalized. Verifique o IP, porta e firewall.';
      }
      return isOllamaConnected.value;
    } finally {
      isApplyingServer.value = false;
    }
  }

  /// Reverts any unpersisted server URL changes when canceling.
  void discardChanges() {
    if (ollamaBaseUrl.value != _initialServerUrl) {
      _repository.updateOllamaBaseUrl(_initialServerUrl);
    }
  }

  /// Persists settings changes and builds the result to return to chat.
  Future<SettingsResult> saveSettings({
    String? pendingHostInput,
  }) async {
    if (isEditingServer.value &&
        pendingHostInput != null &&
        pendingHostInput.trim().isNotEmpty) {
      final normalized = normalizeOllamaUrl(pendingHostInput);
      if (normalized != ollamaBaseUrl.value) {
        await applyServerHost(normalized);
      }
    }

    if (ollamaBaseUrl.value != _initialServerUrl) {
      await _repository.saveOllamaBaseUrl(ollamaBaseUrl.value);
      _initialServerUrl = ollamaBaseUrl.value;
    }

    try {
      await _repository.saveContextMode(contextMode.value);
      await _repository.saveManualContextLength(manualContextLength.value);
    } catch (_) {
      // Ignore storage persistence errors on save.
    }

    return SettingsResult(
      selectedModelName: selectedModelName.value,
      ollamaBaseUrl: ollamaBaseUrl.value,
      availableModels: List<OllamaModel>.from(availableModels.value),
      isOllamaConnected: isOllamaConnected.value,
      isModelLoaded: isModelLoaded.value,
      contextMode: contextMode.value,
      manualContextLength: manualContextLength.value,
    );
  }
}
