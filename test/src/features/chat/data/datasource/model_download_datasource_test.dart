import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:path_provider_platform_interface/path_provider_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/data/datasource/model_download_datasource.dart';

class MockDio extends Mock implements Dio {}

class MockPathProviderPlatform extends Mock
    with MockPlatformInterfaceMixin
    implements PathProviderPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MockDio mockDio;
  late DioClient dioClient;
  late MockPathProviderPlatform mockPathProviderPlatform;
  late ModelDownloadDatasourceImpl datasource;
  late Directory tempDir;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('gemma_test_');

    mockDio = MockDio();
    dioClient = DioClient(mockDio);

    mockPathProviderPlatform = MockPathProviderPlatform();
    PathProviderPlatform.instance = mockPathProviderPlatform;

    when(() => mockPathProviderPlatform.getApplicationDocumentsPath())
        .thenAnswer((_) async => tempDir.path);

    datasource = ModelDownloadDatasourceImpl(dioClient);
  });

  tearDown(() async {
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  group('ModelDownloadDatasourceImpl', () {
    test('getModelPath should append filename to application documents path', () async {
      final path = await datasource.getModelPath('test_model.bin');

      expect(path, '${tempDir.path}/test_model.bin');
    });

    test('modelExists should return false when file does not exist', () async {
      final exists = await datasource.modelExists('non_existing_file.litertlm');

      expect(exists, isFalse);
    });

    test('modelExists should return true when file exists', () async {
      final file = File('${tempDir.path}/existing_file.litertlm');
      await file.writeAsString('mock content');

      final exists = await datasource.modelExists('existing_file.litertlm');

      expect(exists, isTrue);
    });

    test('downloadModel should not call dio when file already exists', () async {
      final file = File('${tempDir.path}/${AppConstants.model2bFileName}');
      await file.writeAsString('existing weights');

      final statuses = <String>[];
      final progresses = <double>[];

      final resultPath = await datasource.downloadModel(
        is2B: true,
        onProgress: (progress, status) {
          progresses.add(progress);
          statuses.add(status);
        },
      );

      expect(resultPath, file.path);
      expect(progresses, [1.0]);
      expect(statuses.first, contains('já está disponível localmente'));
      verifyNever(
        () => mockDio.download(
          any(),
          any(),
          onReceiveProgress: any(named: 'onReceiveProgress'),
        ),
      );
    });

    test('downloadModel should call dio download with 2B url when is2B is true', () async {
      final expectedPath = '${tempDir.path}/${AppConstants.model2bFileName}';

      when(
        () => mockDio.download(
          AppConstants.model2bUrl,
          expectedPath,
          onReceiveProgress: any(named: 'onReceiveProgress'),
        ),
      ).thenAnswer((invocation) async {
        final onReceiveProgress = invocation.namedArguments[#onReceiveProgress] as ProgressCallback?;
        // Simula progresso 50%
        onReceiveProgress?.call(10 * 1024 * 1024, 20 * 1024 * 1024);
        return Response(requestOptions: RequestOptions(path: AppConstants.model2bUrl));
      });

      final statuses = <String>[];
      final progresses = <double>[];

      final resultPath = await datasource.downloadModel(
        is2B: true,
        onProgress: (progress, status) {
          progresses.add(progress);
          statuses.add(status);
        },
      );

      expect(resultPath, expectedPath);
      expect(statuses, contains('Iniciando download...'));
      expect(progresses, contains(0.5));
      verify(
        () => mockDio.download(
          AppConstants.model2bUrl,
          expectedPath,
          onReceiveProgress: any(named: 'onReceiveProgress'),
        ),
      ).called(1);
    });

    test('downloadModel should call dio download with 4B url when is2B is false', () async {
      final expectedPath = '${tempDir.path}/${AppConstants.model4bFileName}';

      when(
        () => mockDio.download(
          AppConstants.model4bUrl,
          expectedPath,
          onReceiveProgress: any(named: 'onReceiveProgress'),
        ),
      ).thenAnswer((invocation) async {
        return Response(requestOptions: RequestOptions(path: AppConstants.model4bUrl));
      });

      final resultPath = await datasource.downloadModel(
        is2B: false,
        onProgress: (_, _) {},
      );

      expect(resultPath, expectedPath);
      verify(
        () => mockDio.download(
          AppConstants.model4bUrl,
          expectedPath,
          onReceiveProgress: any(named: 'onReceiveProgress'),
        ),
      ).called(1);
    });
  });
}
