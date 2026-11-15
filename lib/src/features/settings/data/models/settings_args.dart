import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';

/// Initial state passed when navigating to the settings screen.
class SettingsArgs {
  final String selectedModelName;
  final ResponseStyle responseStyle;
  final String ollamaBaseUrl;
  final List<OllamaModel> availableModels;
  final bool isOllamaConnected;
  final bool isModelLoaded;

  const SettingsArgs({
    this.selectedModelName = '',
    this.responseStyle = ResponseStyle.simple,
    this.ollamaBaseUrl = '',
    this.availableModels = const [],
    this.isOllamaConnected = true,
    this.isModelLoaded = false,
  });
}

/// Result returned when the user saves changes on the settings screen.
class SettingsResult {
  final String selectedModelName;
  final ResponseStyle responseStyle;
  final String ollamaBaseUrl;
  final List<OllamaModel> availableModels;
  final bool isOllamaConnected;
  final bool isModelLoaded;

  const SettingsResult({
    required this.selectedModelName,
    required this.responseStyle,
    required this.ollamaBaseUrl,
    required this.availableModels,
    required this.isOllamaConnected,
    required this.isModelLoaded,
  });
}
