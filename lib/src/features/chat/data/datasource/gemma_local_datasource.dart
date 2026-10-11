import 'package:flutter_gemma/flutter_gemma.dart';
import '../../../../core/constants/app_constants.dart';

abstract class GemmaLocalDatasource {
  Future<void> initialize();
  Future<void> installModelFromFile(String filePath);
  Future<void> createActiveModel({required bool useCpu});
  Future<void> createChat({String? systemInstruction});
  Future<void> addQueryChunk(Message message);
  Stream<dynamic> generateChatResponseAsync();
  void stopGeneration();
  void close();
}

class GemmaLocalDatasourceImpl implements GemmaLocalDatasource {
  InferenceModel? _activeModel;
  InferenceChat? _activeChat;

  @override
  Future<void> initialize() async {
    await FlutterGemma.initialize();
  }

  @override
  Future<void> installModelFromFile(String filePath) async {
    await FlutterGemma.installModel(
      modelType: ModelType.gemmaIt,
    ).fromFile(filePath).install();
  }

  @override
  Future<void> createActiveModel({required bool useCpu}) async {
    final preferredBackend = useCpu ? PreferredBackend.cpu : PreferredBackend.gpu;
    _activeModel = await FlutterGemma.getActiveModel(
      maxTokens: AppConstants.maxTokens,
      preferredBackend: preferredBackend,
      supportImage: false,
    );
  }

  @override
  Future<void> createChat({String? systemInstruction}) async {
    final model = _activeModel;
    if (model == null) {
      throw StateError('Modelo ativo do Gemma ainda não foi inicializado.');
    }
    _activeChat = await model.createChat(
      supportImage: false,
      systemInstruction: systemInstruction ?? AppConstants.systemInstruction,
    );
  }

  @override
  Future<void> addQueryChunk(Message message) async {
    final chat = _activeChat;
    if (chat == null) {
      throw StateError('Chat do Gemma não foi inicializado.');
    }
    await chat.addQueryChunk(message);
  }

  @override
  Stream<dynamic> generateChatResponseAsync() {
    final chat = _activeChat;
    if (chat == null) {
      throw StateError('Chat do Gemma não foi inicializado.');
    }
    return chat.generateChatResponseAsync();
  }

  @override
  void stopGeneration() {
    _activeChat?.stopGeneration();
  }

  @override
  void close() {
    _activeModel?.close();
    _activeModel = null;
    _activeChat = null;
  }
}
