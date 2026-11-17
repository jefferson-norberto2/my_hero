import 'package:flutter/material.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/features/settings/data/services/model_download_service.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Default [SettingsRepository] implementation delegating to remote and local datasources.
class SettingsRepositoryImpl implements SettingsRepository {
  final OllamaRemoteDatasource _ollamaDatasource;
  final SettingsDatasource _settingsDatasource;
  final ModelDownloadService _modelDownloadService;

  SettingsRepositoryImpl({
    required OllamaRemoteDatasource ollamaDatasource,
    required SettingsDatasource settingsDatasource,
    ModelDownloadService? modelDownloadService,
  })  : _ollamaDatasource = ollamaDatasource,
        _settingsDatasource = settingsDatasource,
        _modelDownloadService = modelDownloadService ??
            ModelDownloadService(ollamaDatasource: ollamaDatasource);

  @override
  String get ollamaBaseUrl => _ollamaDatasource.baseUrl;

  @override
  void updateOllamaBaseUrl(String url) {
    _ollamaDatasource.updateBaseUrl(url);
  }

  @override
  Future<String> loadOllamaBaseUrl() async {
    final url = await _settingsDatasource.getOllamaBaseUrl();
    _ollamaDatasource.updateBaseUrl(url);
    return url;
  }

  @override
  Future<void> saveOllamaBaseUrl(String url) async {
    _ollamaDatasource.updateBaseUrl(url);
    await _settingsDatasource.saveOllamaBaseUrl(url);
  }

  @override
  Future<ContextMode> loadContextMode() {
    return _settingsDatasource.getContextMode();
  }

  @override
  Future<void> saveContextMode(ContextMode mode) {
    return _settingsDatasource.saveContextMode(mode);
  }

  @override
  Future<int> loadManualContextLength() {
    return _settingsDatasource.getManualContextLength();
  }

  @override
  Future<void> saveManualContextLength(int length) {
    return _settingsDatasource.saveManualContextLength(length);
  }

  @override
  Future<double?> loadServerMemoryLimitGb() {
    return _settingsDatasource.getServerMemoryLimitGb();
  }

  @override
  Future<void> saveServerMemoryLimitGb(double? limitGb) {
    return _settingsDatasource.saveServerMemoryLimitGb(limitGb);
  }

  @override
  Future<ThemeMode> loadThemeMode() {
    return _settingsDatasource.getThemeMode();
  }

  @override
  Future<void> saveThemeMode(ThemeMode mode) {
    return _settingsDatasource.saveThemeMode(mode);
  }

  @override
  Future<bool> checkOllamaHealth([String? customUrl]) {
    return _ollamaDatasource.checkHealth(customUrl);
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() {
    return _ollamaDatasource.getAvailableModels();
  }

  @override
  Future<int> getModelMaxContextLength(String modelName) {
    return _ollamaDatasource.getModelMaxContextLength(modelName);
  }

  @override
  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  }) async {
    onProgress(0.2, 'Conectando ao Ollama...');
    final isHealthy = await _ollamaDatasource.checkHealth();
    if (!isHealthy) {
      throw StateError(
        'Ollama não está respondendo em ${_ollamaDatasource.baseUrl}.',
      );
    }

    onProgress(0.5, 'Carregando "$modelName" na memória...');
    await _ollamaDatasource.preloadModel(modelName);
    onProgress(1.0, 'Modelo "$modelName" pronto!');
  }

  @override
  ModelDownloadState get currentDownloadState =>
      _modelDownloadService.currentState;

  @override
  Stream<ModelDownloadState> get downloadStateStream =>
      _modelDownloadService.stateStream;

  @override
  Future<ModelDownloadState> downloadModel(String modelName) {
    return _modelDownloadService.downloadModel(modelName);
  }

  @override
  Future<void> cancelModelDownload() {
    return _modelDownloadService.cancelDownload();
  }
}
