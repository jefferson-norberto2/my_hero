import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Repository contract for managing Ollama server settings and model loading.
abstract class SettingsRepository {
  String get ollamaBaseUrl;
  void updateOllamaBaseUrl(String url);
  Future<String> loadOllamaBaseUrl();
  Future<void> saveOllamaBaseUrl(String url);
  Future<bool> checkOllamaHealth([String? customUrl]);
  Future<List<OllamaModel>> getAvailableModels();
  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  });
}
