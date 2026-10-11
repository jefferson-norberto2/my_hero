import 'dart:typed_data';
import 'package:flutter_gemma/flutter_gemma.dart';
import '../adapter/chat_message_adapter.dart';
import '../datasource/gemma_local_datasource.dart';
import '../datasource/model_download_datasource.dart';
import 'chat_repository.dart';

class ChatRepositoryImpl implements ChatRepository {
  final GemmaLocalDatasource _gemmaDatasource;
  final ModelDownloadDatasource _downloadDatasource;

  ChatRepositoryImpl({
    required GemmaLocalDatasource gemmaDatasource,
    required ModelDownloadDatasource downloadDatasource,
  })  : _gemmaDatasource = gemmaDatasource,
        _downloadDatasource = downloadDatasource;

  @override
  Future<void> loadModel({
    required bool is2B,
    required bool useCpu,
    required void Function(double progress, String status) onProgress,
  }) async {
    onProgress(0.0, 'Verificando armazenamento local...');
    final filePath = await _downloadDatasource.downloadModel(
      is2B: is2B,
      onProgress: onProgress,
    );

    onProgress(1.0, 'Carregando modelo na memória...');
    await _gemmaDatasource.installModelFromFile(filePath);
    await _gemmaDatasource.createActiveModel(useCpu: useCpu);
    await _gemmaDatasource.createChat();
  }

  @override
  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
  }) async* {
    final message = ChatMessageAdapter.toGemmaMessage(
      ChatMessage(text: text, isUser: true, imageBytes: imageBytes),
    );

    await _gemmaDatasource.addQueryChunk(message);
    final responseStream = _gemmaDatasource.generateChatResponseAsync();

    await for (final chunk in responseStream) {
      if (chunk is TextResponse) {
        yield chunk.token;
      }
    }
  }

  @override
  void stopGeneration() {
    _gemmaDatasource.stopGeneration();
  }

  @override
  void reset() {
    _gemmaDatasource.close();
  }
}
