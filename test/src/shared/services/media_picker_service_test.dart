import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';

class MockImagePicker extends Mock implements ImagePicker {}
class FakeXFile extends Fake implements XFile {
  final Uint8List _bytes;
  FakeXFile(this._bytes);

  @override
  Future<Uint8List> readAsBytes() async => _bytes;
}

void main() {
  late MockImagePicker mockPicker;
  late MediaPickerService service;

  setUp(() {
    mockPicker = MockImagePicker();
    service = MediaPickerService(picker: mockPicker);
  });

  group('MediaPickerService', () {
    test('pickImageFromGallery should return bytes when file selected', () async {
      final fakeBytes = Uint8List.fromList([1, 2, 3]);
      when(
        () => mockPicker.pickImage(
          source: ImageSource.gallery,
          imageQuality: 80,
        ),
      ).thenAnswer((_) async => FakeXFile(fakeBytes));

      final result = await service.pickImageFromGallery();

      expect(result, fakeBytes);
      verify(
        () => mockPicker.pickImage(
          source: ImageSource.gallery,
          imageQuality: 80,
        ),
      ).called(1);
    });

    test('pickImageFromCamera should return bytes when photo captured', () async {
      final fakeBytes = Uint8List.fromList([4, 5, 6]);
      when(
        () => mockPicker.pickImage(
          source: ImageSource.camera,
          imageQuality: 80,
        ),
      ).thenAnswer((_) async => FakeXFile(fakeBytes));

      final result = await service.pickImageFromCamera();

      expect(result, fakeBytes);
      verify(
        () => mockPicker.pickImage(
          source: ImageSource.camera,
          imageQuality: 80,
        ),
      ).called(1);
    });

    test('pickImage should return null when user cancels selection', () async {
      when(
        () => mockPicker.pickImage(
          source: ImageSource.gallery,
          imageQuality: 80,
        ),
      ).thenAnswer((_) async => null);

      final result = await service.pickImage(source: ImageSource.gallery);

      expect(result, isNull);
    });
  });
}
