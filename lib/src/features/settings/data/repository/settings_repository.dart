import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Repository contract for managing Ollama server, context, model loading, and downloads.
abstract class SettingsRepository {
  String get ollamaBaseUrl;
  void updateOllamaBaseUrl(String url);
  Future<String> loadOllamaBaseUrl();
  Future<void> saveOllamaBaseUrl(String url);
  Future<ContextMode> loadContextMode();
  Future<void> saveContextMode(ContextMode mode);
  Future<int> loadManualContextLength();
  Future<void> saveManualContextLength(int length);
  Future<double?> loadServerMemoryLimitGb();
  Future<void> saveServerMemoryLimitGb(double? limitGb);
  Future<bool> checkOllamaHealth([String? customUrl]);
  Future<List<OllamaModel>> getAvailableModels();
  Future<int> getModelMaxContextLength(String modelName);
  Future<void> loadModel({
    required String modelName,
    required void Function(double progress, String status) onProgress,
  });
  ModelDownloadState get currentDownloadState;
  Stream<ModelDownloadState> get downloadStateStream;
  Future<ModelDownloadState> downloadModel(String modelName);
  Future<void> cancelModelDownload();
}
