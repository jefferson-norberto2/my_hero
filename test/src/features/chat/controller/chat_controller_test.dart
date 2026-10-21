import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockChatRepository extends Mock implements ChatRepository {}
class MockMediaPickerService extends Mock implements MediaPickerService {}

void main() {
  late MockChatRepository mockRepository;
  late MockMediaPickerService mockMediaPicker;
  late ChatController controller;

  setUp(() {
    mockRepository = MockChatRepository();
    mockMediaPicker = MockMediaPickerService();

    when(() => mockRepository.getConversations()).thenAnswer((_) async => []);
    when(() => mockRepository.saveConversations(any())).thenAnswer((_) async {});
    when(() => mockRepository.reset()).thenReturn(null);

    controller = ChatController(
      repository: mockRepository,
      mediaPickerService: mockMediaPicker,
    );
  });

  group('ChatController - Multimodal Image Handling', () {
    test('initial attachedImage should be null', () {
      expect(controller.attachedImage.value, isNull);
    });

    test('setAttachedImage and clearAttachedImage should update state correctly', () {
      final fakeBytes = Uint8List.fromList([1, 2, 3, 4]);

      controller.setAttachedImage(fakeBytes);
      expect(controller.attachedImage.value, fakeBytes);

      controller.clearAttachedImage();
      expect(controller.attachedImage.value, isNull);
    });

    test('pickImage from gallery should store image in attachedImage', () async {
      final fakeBytes = Uint8List.fromList([10, 20, 30]);
      when(() => mockMediaPicker.pickImageFromGallery()).thenAnswer((_) async => fakeBytes);

      await controller.pickImage(fromCamera: false);

      expect(controller.attachedImage.value, fakeBytes);
      verify(() => mockMediaPicker.pickImageFromGallery()).called(1);
    });

    test('pickImage from camera should store image in attachedImage', () async {
      final fakeBytes = Uint8List.fromList([40, 50, 60]);
      when(() => mockMediaPicker.pickImageFromCamera()).thenAnswer((_) async => fakeBytes);

      await controller.pickImage(fromCamera: true);

      expect(controller.attachedImage.value, fakeBytes);
      verify(() => mockMediaPicker.pickImageFromCamera()).called(1);
    });

    test('sendMessage should consume attachedImage and clear it', () async {
      final fakeBytes = Uint8List.fromList([7, 8, 9]);
      controller.isModelLoaded.value = true;
      controller.attachedImage.value = fakeBytes;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['A foto mostra um gato.']));

      await controller.sendMessage('O que tem aqui?');

      expect(controller.attachedImage.value, isNull);
      expect(controller.chatHistory.value.length, 2);
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
      expect(controller.chatHistory.value.first.text, 'O que tem aqui?');
      expect(controller.chatHistory.value.last.text, 'A foto mostra um gato.');

      verify(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: fakeBytes,
        ),
      ).called(1);
    });

    test('sendMessage with empty text but attachedImage should default to prompt', () async {
      final fakeBytes = Uint8List.fromList([11, 22, 33]);
      controller.isModelLoaded.value = true;
      controller.attachedImage.value = fakeBytes;

      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Foto analisada']));

      await controller.sendMessage('');

      expect(controller.attachedImage.value, isNull);
      expect(controller.chatHistory.value.first.text, 'Analise esta imagem');
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
    });

    test('pickAndSendImage should pick image from gallery and send immediately', () async {
      final fakeBytes = Uint8List.fromList([100, 200]);
      controller.isModelLoaded.value = true;

      when(() => mockMediaPicker.pickImageFromGallery()).thenAnswer((_) async => fakeBytes);
      when(
        () => mockRepository.sendMessage(
          text: any(named: 'text'),
          imageBytes: any(named: 'imageBytes'),
        ),
      ).thenAnswer((_) => Stream.fromIterable(['Imagem identificada']));

      await controller.pickAndSendImage('Descreva esta foto');

      expect(controller.chatHistory.value.first.text, 'Descreva esta foto');
      expect(controller.chatHistory.value.first.imageBytes, fakeBytes);
    });
  });
}
