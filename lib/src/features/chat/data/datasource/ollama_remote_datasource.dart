import 'dart:async';
import 'dart:convert';
import 'package:dio/dio.dart';
import '../../../../core/constants/app_constants.dart';
import '../../../../core/network/dio_client.dart';
import '../adapter/ollama_model.dart';

abstract class OllamaRemoteDatasource {
  Future<bool> checkHealth();
  Future<List<OllamaModel>> getAvailableModels();
  Future<List<String>> getLoadedModelNames();
  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName);
  Future<void> preloadModel(String modelName);
  Stream<String> generateChatResponseAsync({
    required String modelName,
    required List<Map<String, dynamic>> messages,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  });
  void stopGeneration();
  void close();
}

class OllamaRemoteDatasourceImpl implements OllamaRemoteDatasource {
  final DioClient _dioClient;
  CancelToken? _currentCancelToken;

  OllamaRemoteDatasourceImpl(this._dioClient);

  String get _baseUrl => AppConstants.ollamaBaseUrl;

  @override
  Future<bool> checkHealth() async {
    try {
      final response = await _dioClient.dio.get(
        '$_baseUrl/',
        options: Options(
          sendTimeout: const Duration(seconds: 3),
          receiveTimeout: const Duration(seconds: 3),
        ),
      );
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<List<Map<String, dynamic>>> _getPsModels() async {
    try {
      final response = await _dioClient.dio.get(
        '$_baseUrl/api/ps',
        options: Options(
          sendTimeout: const Duration(seconds: 5),
          receiveTimeout: const Duration(seconds: 5),
        ),
      );
      if (response.statusCode == 200 && response.data is Map<String, dynamic>) {
        final models = (response.data['models'] as List<dynamic>?) ?? [];
        return models.cast<Map<String, dynamic>>();
      }
      return [];
    } catch (_) {
      return [];
    }
  }

  @override
  Future<List<String>> getLoadedModelNames() async {
    final models = await _getPsModels();
    return models
        .map((m) => (m['name'] as String? ?? m['model'] as String? ?? ''))
        .where((s) => s.isNotEmpty)
        .toList();
  }

  @override
  Future<Map<String, dynamic>?> getLoadedModelInfo(String modelName) async {
    final models = await _getPsModels();
    for (final m in models) {
      final name = m['name'] as String? ?? '';
      final model = m['model'] as String? ?? '';
      if (name == modelName ||
          model == modelName ||
          name.startsWith('$modelName:') ||
          modelName.startsWith('$name:')) {
        return m;
      }
    }
    return null;
  }

  @override
  Future<List<OllamaModel>> getAvailableModels() async {
    final psModels = await _getPsModels();
    final response = await _dioClient.dio.get(
      '$_baseUrl/api/tags',
      options: Options(
        sendTimeout: const Duration(seconds: 5),
        receiveTimeout: const Duration(seconds: 5),
      ),
    );

    if (response.statusCode == 200 && response.data is Map<String, dynamic>) {
      final modelsJson = (response.data['models'] as List<dynamic>?) ?? [];
      return modelsJson.map((m) {
        final map = m as Map<String, dynamic>;
        final name = map['name'] as String? ?? '';
        final model = map['model'] as String? ?? '';

        final loadedMatch = psModels.where(
          (loaded) {
            final lName = loaded['name'] as String? ?? '';
            final lModel = loaded['model'] as String? ?? '';
            return lName == name ||
                lModel == model ||
                lName.startsWith('$name:') ||
                name.startsWith('$lName:');
          },
        ).firstOrNull;

        final isLoaded = loadedMatch != null;
        final sizeVram = (loadedMatch?['size_vram'] as num?)?.toInt() ?? 0;
        final contextLength = (loadedMatch?['context_length'] as num?)?.toInt();

        return OllamaModel.fromJson(
          map,
          isLoaded: isLoaded,
          sizeVram: sizeVram,
          contextLength: contextLength,
        );
      }).toList();
    }
    return [];
  }

  @override
  Future<void> preloadModel(String modelName) async {
    // Chamada sem prompt instrui o Ollama a carregar os pesos na memória/VRAM
    await _dioClient.dio.post(
      '$_baseUrl/api/generate',
      data: {'model': modelName},
      options: Options(
        responseType: ResponseType.json,
        sendTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(minutes: 5),
      ),
    );
  }

  @override
  Stream<String> generateChatResponseAsync({
    required String modelName,
    required List<Map<String, dynamic>> messages,
    void Function(int promptTokens, int evalTokens)? onMetrics,
  }) async* {
    _currentCancelToken?.cancel();
    final cancelToken = CancelToken();
    _currentCancelToken = cancelToken;

    final response = await _dioClient.dio.post<ResponseBody>(
      '$_baseUrl/api/chat',
      data: {
        'model': modelName,
        'messages': messages,
        'stream': true,
      },
      options: Options(
        responseType: ResponseType.stream,
        sendTimeout: const Duration(seconds: 30),
        receiveTimeout: const Duration(minutes: 10),
      ),
      cancelToken: cancelToken,
    );

    final stream = response.data?.stream;
    if (stream == null) return;

    final lineStream = stream
        .cast<List<int>>()
        .transform(utf8.decoder)
        .transform(const LineSplitter());

    await for (final line in lineStream) {
      if (cancelToken.isCancelled) break;
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      try {
        final json = jsonDecode(trimmed) as Map<String, dynamic>;

        if (json['done'] == true) {
          final promptCount = (json['prompt_eval_count'] as num?)?.toInt() ?? 0;
          final evalCount = (json['eval_count'] as num?)?.toInt() ?? 0;
          if (promptCount > 0 || evalCount > 0) {
            onMetrics?.call(promptCount, evalCount);
          }
        }

        final message = json['message'] as Map<String, dynamic>?;
        final content = message?['content'] as String?;
        if (content != null && content.isNotEmpty) {
          yield content;
        }
      } catch (_) {
        // Ignora chunks mal formatados
      }
    }
  }

  @override
  void stopGeneration() {
    _currentCancelToken?.cancel('Interrompido pelo usuário');
    _currentCancelToken = null;
  }

  @override
  void close() {
    stopGeneration();
  }
}
