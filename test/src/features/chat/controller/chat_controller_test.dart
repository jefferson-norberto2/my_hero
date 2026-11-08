import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/ollama_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUpAll(() {
    registerFallbackValue(ResponseStyle.simple);
  });

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenReturn(null);
    when(() => mockRepository.checkOllamaHealth()).thenAnswer((_) async => true);
    when(() => mockRepository.getAvailableModels()).thenAnswer((_) async => [
      const OllamaModel(
        name: 'llama3.2:3b',
        model: 'llama3.2:3b',
        size: 2000000000,
        parameterSize: '3.2B',
      ),
    ]);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
    );
    controller.selectedModelName.value = 'llama3.2:3b';
  });

  group('ChatController - Attachments (Image, Audio, Document)', () {
    test('initial attachedAttachment should be null', () {
      expect(controller.attachedAttachment.value, isNull);
    });

    test('setAttachment and clearAttachment should update state correctly', () {
      final fakeBytes = Uint8List.fromList([1, 2, 3, 4]);
      final attachment = ChatAttachment(
        name: 'teste.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );

      controller.setAttachment(attachment);
      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedImage.value, fakeBytes);

      controller.clearAttachment();
      expect(controller.attachedAttachment.value, isNull);
      expect(controller.attachedImage.value, isNull);
    });

    test('pickImage from gallery should store image in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([10, 20, 30]);
      final attachment = ChatAttachment(
        name: 'foto.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );
      when(() => mockMediaPicker.pickImageFromGallery()).thenAnswer((_) async => attachment);

      await controller.pickImage(fromCamera: false);

      expect(controller.attachedAttachment.value, attachment);
      verify(() => mockMediaPicker.pickImageFromGallery()).called(1);
    });

    test('pickAudio should store audio in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([50, 60]);
      final attachment = ChatAttachment(
        name: 'gravacao.mp3',
        type: AttachmentType.audio,
        bytes: fakeBytes,
      );
      when(() => mockMediaPicker.pickAudioFile()).thenAnswer((_) async => attachment);

      await controller.pickAudio();

      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedAttachment.value?.type, AttachmentType.audio);
      verify(() => mockMediaPicker.pickAudioFile()).called(1);
    });

    test('pickDocument should store document with textContent in attachedAttachment', () async {
      final fakeBytes = Uint8List.fromList([70, 80]);
      final attachment = ChatAttachment(
        name: 'script.dart',
        type: AttachmentType.document,
        bytes: fakeBytes,
        textContent: 'void main() => print("hi");',
      );
      when(() => mockMediaPicker.pickDocumentFile()).thenAnswer((_) async => attachment);

      await controller.pickDocument();

      expect(controller.attachedAttachment.value, attachment);
      expect(controller.attachedAttachment.value?.type, AttachmentType.document);
      expect(controller.attachedAttachment.value?.textContent, 'void main() => print("hi");');
      verify(() => mockMediaPicker.pickDocumentFile()).called(1);
    });

    test('sendMessage should consume attachedAttachment and clear it', () async {
      final fakeBytes = Uint8List.fromList([7, 8, 9]);
      final attachment = ChatAttachment(
        name: 'foto.png',
        type: AttachmentType.image,
        bytes: fakeBytes,
      );
      controller.isModelLoaded.value = true;
      controller.setAttachment(attachment);

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['A foto mostra um gato.']));

      await controller.sendMessage('O que tem aqui?');

      expect(controller.attachedAttachment.value, isNull);
      expect(controller.chatHistory.value.length, 2);
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
      expect(controller.chatHistory.value.first.text, 'O que tem aqui?');
      expect(controller.chatHistory.value.last.text, 'A foto mostra um gato.');

      verify(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: fakeBytes,
          attachment: attachment,
          modelName: 'llama3.2:3b',
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
        ),
      ).called(1);
    });

    test('sendMessage with empty text but document attachment defaults prompt', () async {
      final fakeBytes = Uint8List.fromList([11, 22, 33]);
      final docAttachment = ChatAttachment(
        name: 'exemplo.py',
        type: AttachmentType.document,
        bytes: fakeBytes,
        textContent: 'print(42)',
      );
      controller.isModelLoaded.value = true;
      controller.setAttachment(docAttachment);

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
          attachment: any(named: 'attachment'),
          modelName: any(named: 'modelName'),
          history: any(named: 'history'),
          responseStyle: any(named: 'responseStyle'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Código analisado']));

      await controller.sendMessage('');

      expect(controller.attachedAttachment.value, isNull);
      expect(controller.chatHistory.value.first.text, 'Analise este documento/código');
      expect(controller.chatHistory.value.first.attachment, docAttachment);
    });
  });

  group('ChatController - Ollama Models', () {
    test('refreshModels populates availableModels when healthy', () async {
      await controller.refreshModels();

      expect(controller.isOllamaConnected.value, isTrue);
      expect(controller.availableModels.value.length, 1);
      expect(controller.availableModels.value.first.name, 'llama3.2:3b');
    });

    test('loadModel requests preload on repository', () async {
      when(
        () => mockRepository.loadModel(
          modelName: any(named: 'modelName'),
          onProgress: any(named: 'onProgress'),
        ),
      ).thenAnswer((_) async {});

      await controller.loadModel('llama3.2:3b');

      expect(controller.isModelLoaded.value, isTrue);
      expect(controller.selectedModelName.value, 'llama3.2:3b');
      verify(
        () => mockRepository.loadModel(
          modelName: 'llama3.2:3b',
          onProgress: any(named: 'onProgress'),
        ),
      ).called(1);
    });
  });
}
