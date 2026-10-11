import 'dart:typed_data';

abstract class ChatRepository {
  Future<void> loadModel({
    required bool is2B,
    required bool useCpu,
    required void Function(double progress, String status) onProgress,
  });

  Stream<String> sendMessage({
    required String text,
    Uint8List? imageBytes,
  });

  void stopGeneration();

  void reset();
}
