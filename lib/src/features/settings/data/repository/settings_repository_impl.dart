import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/settings/data/datasource/settings_datasource.dart';
import 'package:my_hero/src/features/settings/data/repository/settings_repository.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Default [SettingsRepository] implementation delegating to remote and local datasources.
class SettingsRepositoryImpl implements SettingsRepository {
  final OllamaRemoteDatasource _ollamaDatasource;
  final SettingsDatasource _settingsDatasource;

  SettingsRepositoryImpl({
    required OllamaRemoteDatasource ollamaDatasource,
    required SettingsDatasource settingsDatasource,
  })  : _ollamaDatasource = ollamaDatasource,
        _settingsDatasource = settingsDatasource;

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
  Future<bool> checkOllamaHealth([String? customUrl]) {
    return _ollamaDatasource.checkHealth(customUrl);
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() {
    return _ollamaDatasource.getAvailableModels();
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
}
