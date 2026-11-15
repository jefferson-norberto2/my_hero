import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/core/utils/url_utils.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';
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
  final responseStyle = signal<ResponseStyle>(ResponseStyle.simple);
  final isOllamaConnected = signal<bool>(true);
  final isModelLoaded = signal<bool>(false);

  final isRefreshingModels = signal<bool>(false);
  final isLoadingModel = signal<bool>(false);
  final loadingProgress = signal<double>(0.0);
  final loadingStatus = signal<String>('');
  final loadErrorMessage = signal<String?>(null);

  final isEditingServer = signal<bool>(false);
  final isTestingConnection = signal<bool>(false);
  final isApplyingServer = signal<bool>(false);
  final connectionTestSuccess = signal<bool?>(null);
  final connectionTestMessage = signal<String?>(null);

  String _initialServerUrl = AppConstants.ollamaBaseUrl;

  /// Initializes the controller with optional route arguments or loads from storage.
  Future<void> init([SettingsArgs? args]) async {
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
      responseStyle.value = args.responseStyle;
      isOllamaConnected.value = args.isOllamaConnected;
      isModelLoaded.value = args.isModelLoaded ||
          availableModels.value.any(
            (m) => m.name == selectedModelName.value && m.isLoaded,
          );
      return;
    }

    _initialServerUrl = _repository.ollamaBaseUrl;
    ollamaBaseUrl.value = _initialServerUrl;
    await loadInitialData();
  }

  /// Loads the persisted server URL and fetches available models.
  Future<void> loadInitialData() async {
    try {
      final savedUrl = await _repository.loadOllamaBaseUrl();
      _initialServerUrl = savedUrl;
      ollamaBaseUrl.value = savedUrl;
    } catch (_) {
      // Keep default server URL if loading from disk fails.
    }
    await refreshModels();
  }

  void selectModel(String modelName) {
    selectedModelName.value = modelName;
    loadErrorMessage.value = null;
    isModelLoaded.value = availableModels.value.any(
      (m) => m.name == modelName && m.isLoaded,
    );
  }

  void selectResponseStyle(ResponseStyle style) {
    responseStyle.value = style;
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

  /// Persists the server URL if changed and builds the result to return to chat.
  Future<SettingsResult> saveSettings({String? pendingHostInput}) async {
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

    return SettingsResult(
      selectedModelName: selectedModelName.value,
      responseStyle: responseStyle.value,
      ollamaBaseUrl: ollamaBaseUrl.value,
      availableModels: List<OllamaModel>.from(availableModels.value),
      isOllamaConnected: isOllamaConnected.value,
      isModelLoaded: isModelLoaded.value,
    );
  }
}
