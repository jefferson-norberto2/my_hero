import 'dart:typed_data';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../shared/services/media_picker_service.dart';
import '../data/adapter/chat_message_adapter.dart';
import '../data/adapter/conversation_model.dart';
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
  final is2BModel = signal<bool>(true);
  final chatHistory = signal<List<ChatMessage>>([]);
  final errorMessage = signal<String?>(null);

  // Histórico de conversas
  final conversations = signal<List<Conversation>>([]);
  final currentConversationId = signal<String?>(null);

  bool _stopGenerationFlag = false;

  bool get isCpu => backendsSelected.value[0];

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

  /// Cria uma nova conversa com contexto 100% limpo e isolado
  Future<void> createNewConversation() async {
    final now = DateTime.now();
    final newConv = Conversation(
      id: now.millisecondsSinceEpoch.toString(),
      title: 'Nova Conversa',
      messages: [],
      is2B: is2BModel.value,
      useCpu: isCpu,
      createdAt: now,
      updatedAt: now,
    );

    conversations.value = [newConv, ...conversations.value];
    currentConversationId.value = newConv.id;
    chatHistory.value = [];

    // Se o modelo já estiver carregado, inicia uma nova sessão de chat no Gemma
    // sem nenhum histórico, garantindo que o modelo não misture o contexto anterior!
    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(contextMessages: null);
      } catch (_) {}
    }

    await _saveCurrentState();
  }

  /// Seleciona uma conversa e carrega exclusivamente o contexto dela
  Future<void> selectConversation(String id) async {
    if (isGenerating.value) return;

    final conv = conversations.value.where((c) => c.id == id).firstOrNull;
    if (conv == null) return;

    currentConversationId.value = conv.id;
    chatHistory.value = List.from(conv.messages);
    is2BModel.value = conv.is2B;
    backendsSelected.value = [conv.useCpu, !conv.useCpu];

    // Se o modelo estiver carregado, reinicia a sessão de chat e sincroniza
    // APENAS as mensagens da conversa selecionada
    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(contextMessages: conv.messages);
      } catch (_) {}
    }
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

  void selectBackend(int index) {
    if (isLoading.value) return;
    final updated = List<bool>.generate(
      backendsSelected.value.length,
      (i) => i == index,
    );
    backendsSelected.value = updated;

    final conv = currentConversation;
    if (conv != null) {
      conv.useCpu = isCpu;
      _saveCurrentState();
    }
  }

  Future<void> downloadAndLoadModel({
    bool is2B = true,
    List<ChatMessage>? contextMessages,
  }) async {
    isLoading.value = true;
    downloadProgress.value = 0.0;
    loadingStatus.value = 'Verificando armazenamento local...';
    errorMessage.value = null;

    final fileName = is2B ? 'gemma_2b.litertlm' : 'gemma_4b.litertlm';

    try {
      final messagesToSync = contextMessages ?? chatHistory.value;

      await _repository.loadModel(
        is2B: is2B,
        useCpu: isCpu,
        onProgress: (progress, status) {
          downloadProgress.value = progress;
          loadingStatus.value = status;
        },
        contextMessages: messagesToSync,
      );

      is2BModel.value = is2B;
      isModelLoaded.value = true;
      isLoading.value = false;

      final conv = currentConversation;
      if (conv != null) {
        conv.is2B = is2B;
        conv.useCpu = isCpu;
        _saveCurrentState();
      }

      final contextNote = (messagesToSync.isNotEmpty)
          ? ' Contexto da conversa sincronizado!'
          : '';

      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Modelo $fileName carregado com sucesso!$contextNote',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    } catch (e) {
      isLoading.value = false;
      errorMessage.value = 'Falha ao carregar o modelo: $e';
    }
  }

  /// Altera o modelo da conversa atual preservando apenas o contexto das mensagens desta conversa
  Future<void> switchModelWithContext({
    required bool is2B,
    required bool useCpu,
  }) async {
    if (isLoading.value || isGenerating.value) return;

    backendsSelected.value = [useCpu, !useCpu];
    await downloadAndLoadModel(
      is2B: is2B,
      contextMessages: chatHistory.value,
    );
  }

  Future<void> sendMessage(String text, [Uint8List? imageBytes]) async {
    final trimmedText = text.trim();
    if (trimmedText.isEmpty || !isModelLoaded.value || isGenerating.value) return;

    _stopGenerationFlag = false;
    isGenerating.value = true;

    // Atualiza o título se for uma conversa nova
    final conv = currentConversation;
    if (conv != null && (conv.title == 'Nova Conversa' || conv.messages.isEmpty)) {
      conv.title = trimmedText.length > 25
          ? '${trimmedText.substring(0, 25)}...'
          : trimmedText;
    }

    // Adiciona a pergunta do usuário e uma resposta vazia para preenchimento em stream
    chatHistory.value = [
      ...chatHistory.value,
      ChatMessage(text: trimmedText, isUser: true, imageBytes: imageBytes),
      ChatMessage(text: '', isUser: false),
    ];
    _syncMessagesToCurrentConversation();

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
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
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
    isModelLoaded.value = false;
  }

  void dispose() {
    _repository.reset();
  }
}
