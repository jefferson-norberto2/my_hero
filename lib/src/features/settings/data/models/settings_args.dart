import 'package:my_hero/src/shared/models/context_settings.dart';
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
  final ContextMode contextMode;
  final int manualContextLength;

  const SettingsArgs({
    this.selectedModelName = '',
    this.responseStyle = ResponseStyle.simple,
    this.ollamaBaseUrl = '',
    this.availableModels = const [],
    this.isOllamaConnected = true,
    this.isModelLoaded = false,
    this.contextMode = ContextMode.automatic,
    this.manualContextLength = ContextSettings.fallbackMaxContext,
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
  final ContextMode contextMode;
  final int manualContextLength;

  const SettingsResult({
    required this.selectedModelName,
    required this.responseStyle,
    required this.ollamaBaseUrl,
    required this.availableModels,
    required this.isOllamaConnected,
    required this.isModelLoaded,
    this.contextMode = ContextMode.automatic,
    this.manualContextLength = ContextSettings.fallbackMaxContext,
  });
}
