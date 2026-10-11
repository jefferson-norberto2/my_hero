import 'dart:typed_data';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../shared/services/media_picker_service.dart';
import '../data/adapter/chat_message_adapter.dart';
import '../data/repository/chat_repository.dart';

class ChatController {
  final ChatRepository _repository;
  final MediaPickerService _mediaPickerService;

  ChatController({
    required ChatRepository repository,
    required MediaPickerService mediaPickerService,
  })  : _repository = repository,
        _mediaPickerService = mediaPickerService;

  // Signals de estado reativo
  final isModelLoaded = signal(false);
  final isLoading = signal(false);
  final isGenerating = signal(false);
  final downloadProgress = signal(0.0);
  final loadingStatus = signal('');
  final backendsSelected = signal<List<bool>>([true, false]); // [CPU, GPU]
  final chatHistory = signal<List<ChatMessage>>([]);
  final errorMessage = signal<String?>(null);

  bool _stopGenerationFlag = false;

  bool get isCpu => backendsSelected.value[0];

  void selectBackend(int index) {
    if (isLoading.value) return;
    final updated = List<bool>.generate(
      backendsSelected.value.length,
      (i) => i == index,
    );
    backendsSelected.value = updated;
  }

  Future<void> downloadAndLoadModel([bool is2B = true]) async {
    isLoading.value = true;
    downloadProgress.value = 0.0;
    loadingStatus.value = 'Verificando armazenamento local...';
    errorMessage.value = null;

    final fileName = is2B ? 'gemma_2b.litertlm' : 'gemma_4b.litertlm';

    try {
      await _repository.loadModel(
        is2B: is2B,
        useCpu: isCpu,
        onProgress: (progress, status) {
          downloadProgress.value = progress;
          loadingStatus.value = status;
        },
      );

      isModelLoaded.value = true;
      isLoading.value = false;
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Modelo $fileName carregado com sucesso do dispositivo!',
          isUser: false,
        ),
      ];
    } catch (e) {
      isLoading.value = false;
      errorMessage.value = 'Falha ao carregar o modelo: $e';
    }
  }

  Future<void> sendMessage(String text, [Uint8List? imageBytes]) async {
    final trimmedText = text.trim();
    if (trimmedText.isEmpty || !isModelLoaded.value || isGenerating.value) return;

    _stopGenerationFlag = false;
    isGenerating.value = true;

    // Adiciona a pergunta do usuário e uma resposta vazia para preenchimento em stream
    chatHistory.value = [
      ...chatHistory.value,
      ChatMessage(text: trimmedText, isUser: true, imageBytes: imageBytes),
      ChatMessage(text: '', isUser: false),
    ];

    try {
      final stream = _repository.sendMessage(
        text: trimmedText,
        imageBytes: imageBytes,
      );

      await for (final token in stream) {
        if (_stopGenerationFlag) break;

        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        final lastMsg = currentList[lastIndex];

        currentList[lastIndex] = lastMsg.copyWith(
          text: lastMsg.text + token,
        );
        chatHistory.value = currentList;
      }
    } catch (e) {
      if (!_stopGenerationFlag) {
        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          text: 'Sistema: Erro ao gerar resposta ($e)',
        );
        chatHistory.value = currentList;
      }
    } finally {
      isGenerating.value = false;
    }
  }

  Future<void> pickAndSendImage(String currentText) async {
    try {
      final imageBytes = await _mediaPickerService.pickImageFromGallery();
      if (imageBytes != null) {
        await sendMessage(
          currentText.isEmpty ? 'Analise esta imagem' : currentText,
          imageBytes,
        );
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar imagem ($e)',
          isUser: false,
        ),
      ];
    }
  }

  void stopGeneration() {
    if (isGenerating.value) {
      _stopGenerationFlag = true;
      _repository.stopGeneration();
      isGenerating.value = false;
    }
  }

  void reset() {
    if (isGenerating.value) return;
    _repository.reset();
    isModelLoaded.value = false;
    chatHistory.value = [];
  }

  void dispose() {
    _repository.reset();
  }
}
