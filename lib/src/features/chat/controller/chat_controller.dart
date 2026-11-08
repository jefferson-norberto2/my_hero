import 'dart:typed_data';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../shared/services/media_picker_service.dart';
import '../data/adapter/chat_attachment.dart';
import '../data/adapter/chat_message_adapter.dart';
import '../data/adapter/conversation_model.dart';
import '../data/adapter/ollama_model.dart';
import '../data/adapter/response_style.dart';
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
  final availableModels = signal<List<OllamaModel>>([]);
  final selectedModelName = signal<String>('');
  final isOllamaConnected = signal<bool>(true);
  final isModelLoaded = signal<bool>(false);
  final isLoading = signal<bool>(false);
  final isGenerating = signal<bool>(false);
  final loadingProgress = signal<double>(0.0);
  final loadingStatus = signal<String>('');
  final responseStyle = signal<ResponseStyle>(ResponseStyle.simple);
  final chatHistory = signal<List<ChatMessage>>([]);
  final errorMessage = signal<String?>(null);
  final attachedAttachment = signal<ChatAttachment?>(null);
  final attachedImage = signal<Uint8List?>(null);

  // Retrocompatibilidade para widgets existentes
  bool get isCpu => true;
  final backendsSelected = signal<List<bool>>([true, false]);
  final downloadProgress = signal<double>(0.0);
  final is2BModel = signal<bool>(true);

  // Histórico de conversas
  final conversations = signal<List<Conversation>>([]);
  final currentConversationId = signal<String?>(null);

  bool _stopGenerationFlag = false;

  Conversation? get currentConversation {
    final id = currentConversationId.value;
    if (id == null) return null;
    try {
      return conversations.value.firstWhere((c) => c.id == id);
    } catch (_) {
      return null;
    }
  }

  Future<void> init() async {
    await loadSavedConversations();
    await refreshModels();
  }

  Future<void> refreshModels() async {
    try {
      final isHealthy = await _repository.checkOllamaHealth();
      isOllamaConnected.value = isHealthy;

      if (!isHealthy) {
        errorMessage.value =
            'Ollama não detectado em http://localhost:11434. Verifique se o serviço está ativo.';
        return;
      }

      final models = await _repository.getAvailableModels();
      availableModels.value = models;

      if (models.isNotEmpty) {
        // Se já houver um modelo carregado na memória do Ollama, prioriza ele
        final alreadyLoaded = models.where((m) => m.isLoaded).firstOrNull;

        if (selectedModelName.value.isEmpty) {
          final target = alreadyLoaded?.name ??
              (currentConversation?.modelName.isNotEmpty == true
                  ? currentConversation!.modelName
                  : models.first.name);
          selectedModelName.value = target;
        }

        // Verifica se o modelo atualmente selecionado já está carregado
        final currentModelInfo =
            models.where((m) => m.name == selectedModelName.value).firstOrNull;
        if (currentModelInfo != null && currentModelInfo.isLoaded) {
          isModelLoaded.value = true;
        }
      }
    } catch (e) {
      isOllamaConnected.value = false;
      errorMessage.value = 'Erro ao consultar modelos do Ollama: $e';
    }
  }

  Future<void> loadSavedConversations() async {
    try {
      final saved = await _repository.getConversations();
      if (saved.isNotEmpty) {
        conversations.value = saved;
        await selectConversation(saved.first.id);
      } else {
        await createNewConversation();
      }
    } catch (_) {
      await createNewConversation();
    }
  }

  /// Cria uma nova conversa com contexto 100% limpo
  Future<void> createNewConversation() async {
    final now = DateTime.now();
    final modelToUse = selectedModelName.value.isNotEmpty
        ? selectedModelName.value
        : (availableModels.value.isNotEmpty ? availableModels.value.first.name : '');

    final newConv = Conversation(
      id: now.millisecondsSinceEpoch.toString(),
      title: 'Nova Conversa',
      messages: [],
      modelName: modelToUse,
      responseStyle: responseStyle.value,
      createdAt: now,
      updatedAt: now,
    );

    conversations.value = [newConv, ...conversations.value];
    currentConversationId.value = newConv.id;
    chatHistory.value = [];

    await _saveCurrentState();
  }

  /// Seleciona uma conversa e carrega o histórico e estilo dela
  Future<void> selectConversation(String id) async {
    if (isGenerating.value) return;

    final conv = conversations.value.where((c) => c.id == id).firstOrNull;
    if (conv == null) return;

    currentConversationId.value = conv.id;
    chatHistory.value = List.from(conv.messages);
    if (conv.modelName.isNotEmpty) {
      selectedModelName.value = conv.modelName;
    }
    responseStyle.value = conv.responseStyle;
    is2BModel.value = conv.is2B;
  }

  Future<void> deleteConversation(String id) async {
    if (isGenerating.value) return;

    final updated = conversations.value.where((c) => c.id != id).toList();
    conversations.value = updated;

    if (currentConversationId.value == id) {
      if (updated.isNotEmpty) {
        await selectConversation(updated.first.id);
      } else {
        await createNewConversation();
      }
    }

    await _repository.saveConversations(conversations.value);
  }

  /// Carrega o modelo selecionado na memória do Ollama
  Future<void> loadModel(String modelName) async {
    isLoading.value = true;
    loadingProgress.value = 0.0;
    downloadProgress.value = 0.0;
    loadingStatus.value = 'Solicitando carregamento de $modelName no Ollama...';
    errorMessage.value = null;

    try {
      await _repository.loadModel(
        modelName: modelName,
        onProgress: (progress, status) {
          loadingProgress.value = progress;
          downloadProgress.value = progress;
          loadingStatus.value = status;
        },
      );

      selectedModelName.value = modelName;
      isModelLoaded.value = true;
      isLoading.value = false;

      // Atualiza os modelos disponíveis para refletir que agora está em memória
      await refreshModels();

      final conv = currentConversation;
      if (conv != null) {
        conv.modelName = modelName;
        await _saveCurrentState();
      }

      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Modelo "$modelName" carregado na memória do computador!',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    } catch (e) {
      isLoading.value = false;
      errorMessage.value = 'Falha ao carregar o modelo no Ollama: $e';
    }
  }

  /// Altera as configurações da conversa (modelo e estilo de resposta)
  Future<void> updateConversationSettings({
    required String modelName,
    required ResponseStyle newResponseStyle,
  }) async {
    if (isLoading.value || isGenerating.value) return;

    final modelChanged = modelName != selectedModelName.value;
    final styleChanged = newResponseStyle != responseStyle.value;

    if (!modelChanged && !styleChanged) return;

    responseStyle.value = newResponseStyle;
    selectedModelName.value = modelName;
    is2BModel.value = modelName.contains('2b') || modelName.contains('3b');

    final conv = currentConversation;
    if (conv != null) {
      conv.modelName = modelName;
      conv.responseStyle = newResponseStyle;
      await _saveCurrentState();
    }

    if (modelChanged) {
      await loadModel(modelName);
    } else if (styleChanged) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Estilo de resposta alterado para "${newResponseStyle.label}".',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  void setAttachment(ChatAttachment? attachment) {
    attachedAttachment.value = attachment;
    attachedImage.value =
        attachment?.type == AttachmentType.image ? attachment?.bytes : null;
  }

  void clearAttachment() {
    attachedAttachment.value = null;
    attachedImage.value = null;
  }

  void setAttachedImage(Uint8List? bytes) {
    if (bytes == null) {
      clearAttachment();
    } else {
      setAttachment(ChatAttachment(
        name: 'imagem.png',
        type: AttachmentType.image,
        bytes: bytes,
      ));
    }
  }

  void clearAttachedImage() {
    clearAttachment();
  }

  Future<void> pickImage({bool fromCamera = false}) async {
    try {
      final attachment = fromCamera
          ? await _mediaPickerService.pickImageFromCamera()
          : await _mediaPickerService.pickImageFromGallery();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar imagem ($e)',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    }
  }

  Future<void> pickAudio() async {
    try {
      final attachment = await _mediaPickerService.pickAudioFile();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar arquivo de áudio ($e)',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    }
  }

  Future<void> pickDocument() async {
    try {
      final attachment = await _mediaPickerService.pickDocumentFile();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar documento ($e)',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    }
  }

  Future<void> sendMessage(String text, [ChatAttachment? customAttachment]) async {
    final attachment = customAttachment ?? attachedAttachment.value;
    final trimmedText = text.trim();

    String effectiveText = trimmedText;
    if (effectiveText.isEmpty && attachment != null) {
      switch (attachment.type) {
        case AttachmentType.image:
          effectiveText = 'Analise esta imagem';
          break;
        case AttachmentType.audio:
          effectiveText = 'Analise este áudio';
          break;
        case AttachmentType.document:
          effectiveText = 'Analise este documento/código';
          break;
      }
    }

    final currentModel = selectedModelName.value;
    if (effectiveText.isEmpty || currentModel.isEmpty || isGenerating.value) return;

    isModelLoaded.value = true;
    clearAttachment();
    _stopGenerationFlag = false;
    isGenerating.value = true;

    // Atualiza o título se for uma conversa nova
    final conv = currentConversation;
    if (conv != null && (conv.title == 'Nova Conversa' || conv.messages.isEmpty)) {
      conv.title = effectiveText.length > 25
          ? '${effectiveText.substring(0, 25)}...'
          : effectiveText;
    }

    final userMessage = ChatMessage(
      text: effectiveText,
      isUser: true,
      attachment: attachment,
      imageBytes: attachment?.type == AttachmentType.image ? attachment?.bytes : null,
    );

    // Adiciona a pergunta do usuário e uma resposta vazia para preenchimento em stream
    chatHistory.value = [
      ...chatHistory.value,
      userMessage,
      ChatMessage(text: '', isUser: false),
    ];
    _syncMessagesToCurrentConversation();

    final promptToSend = responseStyle.value.applyToPrompt(effectiveText);

    try {
      final stream = _repository.sendMessage(
        text: promptToSend,
        attachment: attachment,
        imageBytes: attachment?.type == AttachmentType.image ? attachment?.bytes : null,
        modelName: currentModel,
        history: chatHistory.value.sublist(0, chatHistory.value.length - 2),
        responseStyle: responseStyle.value,
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
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  Future<void> pickAndSendImage(String currentText, {bool fromCamera = false}) async {
    try {
      final attachment = fromCamera
          ? await _mediaPickerService.pickImageFromCamera()
          : await _mediaPickerService.pickImageFromGallery();
      if (attachment != null) {
        await sendMessage(
          currentText.isEmpty ? 'Analise esta imagem' : currentText,
          attachment,
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
      _syncMessagesToCurrentConversation();
    }
  }

  void _syncMessagesToCurrentConversation() {
    final conv = currentConversation;
    if (conv != null) {
      conv.messages = List.from(chatHistory.value);
      conv.updatedAt = DateTime.now();
    }
  }

  Future<void> _saveCurrentState() async {
    await _repository.saveConversations(conversations.value);
  }

  void stopGeneration() {
    if (isGenerating.value) {
      _stopGenerationFlag = true;
      _repository.stopGeneration();
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      _saveCurrentState();
    }
  }

  void reset() {
    if (isGenerating.value) return;
    _repository.reset();
  }

  void dispose() {
    _repository.reset();
  }
}
