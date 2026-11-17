import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:path_provider/path_provider.dart';

/// Persists application-wide settings such as the Ollama server URL,
/// context window preferences, and theme mode.
abstract class SettingsDatasource {
  Future<String> getOllamaBaseUrl();
  Future<void> saveOllamaBaseUrl(String url);
  Future<ContextMode> getContextMode();
  Future<void> saveContextMode(ContextMode mode);
  Future<int> getManualContextLength();
  Future<void> saveManualContextLength(int length);
  Future<ThemeMode> getThemeMode();
  Future<void> saveThemeMode(ThemeMode mode);
}

class SettingsDatasourceImpl implements SettingsDatasource {
  static const String _settingsFileName = 'app_settings.json';
  static const String _ollamaUrlKey = 'ollama_base_url';
  static const String _contextModeKey = 'context_mode';
  static const String _manualContextLengthKey = 'manual_context_length';
  static const String _themeModeKey = 'theme_mode';

  Future<File> _getFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_settingsFileName');
  }

  Future<Map<String, dynamic>> _readSettingsMap() async {
    try {
      final file = await _getFile();
      if (!await file.exists()) return {};
      final content = await file.readAsString();
      if (content.trim().isEmpty) return {};
      return jsonDecode(content) as Map<String, dynamic>;
    } catch (_) {
      return {};
    }
  }

  Future<void> _updateSettingsMap(
    void Function(Map<String, dynamic> map) updater,
  ) async {
    try {
      final file = await _getFile();
      final map = await _readSettingsMap();
      updater(map);
      await file.writeAsString(jsonEncode(map));
    } catch (_) {
      // Ignore write failures silently to preserve app stability.
    }
  }

  @override
  Future<String> getOllamaBaseUrl() async {
    final map = await _readSettingsMap();
    final url = map[_ollamaUrlKey] as String?;
    if (url != null && url.trim().isNotEmpty) {
      return url.trim();
    }
    return AppConstants.ollamaBaseUrl;
  }

  @override
  Future<void> saveOllamaBaseUrl(String url) {
    return _updateSettingsMap((map) => map[_ollamaUrlKey] = url);
  }

  @override
  Future<ContextMode> getContextMode() async {
    final map = await _readSettingsMap();
    return ContextMode.fromString(map[_contextModeKey] as String?);
  }

  @override
  Future<void> saveContextMode(ContextMode mode) {
    return _updateSettingsMap((map) => map[_contextModeKey] = mode.name);
  }

  @override
  Future<int> getManualContextLength() async {
    final map = await _readSettingsMap();
    final value = (map[_manualContextLengthKey] as num?)?.toInt();
    if (value != null && value > 0) {
      return value;
    }
    return ContextSettings.fallbackMaxContext;
  }

  @override
  Future<void> saveManualContextLength(int length) {
    return _updateSettingsMap((map) => map[_manualContextLengthKey] = length);
  }

  @override
  Future<ThemeMode> getThemeMode() async {
    final map = await _readSettingsMap();
    final raw = map[_themeModeKey] as String?;
    if (raw == ThemeMode.dark.name) {
      return ThemeMode.dark;
    }
    return ThemeMode.light;
  }

  @override
  Future<void> saveThemeMode(ThemeMode mode) {
    final normalized =
        mode == ThemeMode.dark ? ThemeMode.dark : ThemeMode.light;
    return _updateSettingsMap((map) => map[_themeModeKey] = normalized.name);
  }
}
