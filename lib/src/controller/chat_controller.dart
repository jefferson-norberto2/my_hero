import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:my_hero/src/models/chat_message.dart';
import 'package:path_provider/path_provider.dart';
import 'package:signals/signals.dart';

class ChatController {
  final Dio dio;
  ChatController(this.dio);
  
  final List<bool> backendsSelected = ListSignal<bool>([true, false]);

  final isModelLoaded = signal(false);
  final isLoading = signal(false);
  final isGenerating = signal(false);
  final stopGenerationFlag = signal(false);

  // Download state variables
  final downloadProgress = signal(0.0);
  final loadingStatus = signal('');

  final chatHistory = ListSignal<ChatMessage>([]);
  InferenceModel? activeModel;
  InferenceChat? activeChat;

  Future<bool> downloadAndLoadModel([bool is2B = true]) async {
      isLoading.value = true;
      downloadProgress.value = 0.0;
      loadingStatus.value = 'Checking local storage...';

    // Replace these URLs with your direct download links
    final downloadUrl = is2B
        ? 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm'
        : 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm';

    final fileName = is2B ? 'gemma_2b.litertlm' : 'gemma_4b.litertlm';

    try {
      // 1. Get the app's internal documents directory
      final directory = await getApplicationDocumentsDirectory();
      final filePath = '${directory.path}/$fileName';
      final file = File(filePath);

      // 2. Download the file if it doesn't exist
      if (!await file.exists()) {
        loadingStatus.value = 'Starting download...';

        await dio.download(
          downloadUrl,
          filePath,
          onReceiveProgress: (received, total) {
            if (total != -1) {
                downloadProgress.value = received / total;
                loadingStatus.value =
                    'Downloading: ${(received / 1024 / 1024).toStringAsFixed(1)} MB / ${(total / 1024 / 1024).toStringAsFixed(1)} MB';
            }
          },
        );
      }

      // 3. Load the model from the local file
        loadingStatus.value = 'Loading model $fileName into memory...';
        downloadProgress.value = 1.0;

      await FlutterGemma.installModel(
        modelType: ModelType.gemmaIt,
      ).fromFile(filePath).install();

      var preferredBackend = backendsSelected[0]
          ? PreferredBackend.cpu
          : PreferredBackend.gpu;

      activeModel = await FlutterGemma.getActiveModel(
        maxTokens: 2048,
        preferredBackend: preferredBackend,
        supportImage: true,
      );

      activeChat = await activeModel!.createChat(
        supportImage: true,
        systemInstruction:
            'You are a helpful assistant and always reply in Portuguese.',
      );

        isModelLoaded.value = true;
        isLoading.value = false;
        chatHistory.add(
          ChatMessage(
            text:
                "System: Model $fileName loaded successfully from device storage!",
            isUser: false,
          ),
        );
      return true;
    } catch (e) {
      isLoading.value = false;
    }
    return false;
  }
  
}