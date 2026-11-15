import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../../../../core/constants/app_constants.dart';

abstract class SettingsDatasource {
  Future<String> getOllamaBaseUrl();
  Future<void> saveOllamaBaseUrl(String url);
}

class SettingsDatasourceImpl implements SettingsDatasource {
  static const String _settingsFileName = 'app_settings.json';
  static const String _ollamaUrlKey = 'ollama_base_url';

  Future<File> _getFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_settingsFileName');
  }

  @override
  Future<String> getOllamaBaseUrl() async {
    try {
      final file = await _getFile();
      if (!await file.exists()) {
        return AppConstants.ollamaBaseUrl;
      }
      final content = await file.readAsString();
      if (content.trim().isEmpty) return AppConstants.ollamaBaseUrl;

      final map = jsonDecode(content) as Map<String, dynamic>;
      final url = map[_ollamaUrlKey] as String?;
      if (url != null && url.trim().isNotEmpty) {
        return url.trim();
      }
      return AppConstants.ollamaBaseUrl;
    } catch (_) {
      return AppConstants.ollamaBaseUrl;
    }
  }

  @override
  Future<void> saveOllamaBaseUrl(String url) async {
    try {
      final file = await _getFile();
      Map<String, dynamic> map = {};
      if (await file.exists()) {
        final content = await file.readAsString();
        if (content.trim().isNotEmpty) {
          map = jsonDecode(content) as Map<String, dynamic>;
        }
      }
      map[_ollamaUrlKey] = url;
      await file.writeAsString(jsonEncode(map));
    } catch (_) {
      // Falha silenciosa
    }
  }
}
