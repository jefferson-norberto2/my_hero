import 'dart:io';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/data/datasource/gemma_local_datasource.dart';

class FakeInferenceModel extends Fake implements InferenceModel {
  @override
  PreferredBackend? get activeBackend => PreferredBackend.gpu;

  @override
  PreferredBackend? get activeVisionBackend => PreferredBackend.gpu;

  @override
  Future<void> close() async {}
}

void main() {
  late GemmaLocalDatasourceImpl datasource;

  setUp(() {
    datasource = GemmaLocalDatasourceImpl();
  });

  group('GemmaLocalDatasourceImpl state validation', () {
    test('activeBackend and activeVisionBackend should be null before active model is initialized', () {
      expect(datasource.activeBackend, isNull);
      expect(datasource.activeVisionBackend, isNull);
    });

    test('openChat should throw StateError when active model is not initialized', () async {
      expect(
        () => datasource.openChat(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Modelo ativo do Gemma ainda não foi inicializado'),
          ),
        ),
      );
    });

    test('addQueryChunk should throw StateError when active chat is not initialized', () async {
      final message = Message.text(text: 'Teste', isUser: true);

      expect(
        () => datasource.addQueryChunk(message),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Chat do Gemma não foi inicializado'),
          ),
        ),
      );
    });

    test('generateChatResponseAsync should throw StateError when active chat is not initialized', () {
      expect(
        () => datasource.generateChatResponseAsync(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Chat do Gemma não foi inicializado'),
          ),
        ),
      );
    });

    test('stopGeneration should complete normally even when chat is null', () async {
      await expectLater(datasource.stopGeneration(), completes);
    });

    test('close should complete normally when called without active model or chat', () async {
      await expectLater(datasource.close(), completes);
    });

    test('cancelActiveDownload should not throw when no download is active', () {
      expect(() => datasource.cancelActiveDownload(), returnsNormally);
    });
  });

  group('GemmaLocalDatasourceImpl install & activation safety', () {
    test('activateInstalledModel with restored FileSource spec should skip installer and NEVER call uninstallModel', () async {
      final spec = AppConstants.model2b;
      var uninstallCalled = false;
      var getModelPathCalled = false;
      var activateFromFileCalled = false;

      final restoredSpec = InferenceModelSpec(
        name: spec.fileName,
        modelSource: FileSource('/data/user/0/com.example.my_hero/app_flutter/${spec.fileName}'),
        modelType: spec.modelType,
        fileType: spec.fileType,
      );

      final ds = GemmaLocalDatasourceImpl(
        activeModelSpecFn: () => restoredSpec,
        getModelPathFn: (_) async {
          getModelPathCalled = true;
          return '/tmp/path';
        },
        activateFromFileFn: ({
          required modelType,
          required fileType,
          required filePath,
        }) async {
          activateFromFileCalled = true;
        },
        uninstallModelFn: (_) async {
          uninstallCalled = true;
        },
      );

      await ds.activateInstalledModel(
        fileName: spec.fileName,
        modelType: spec.modelType,
        fileType: spec.fileType,
      );

      expect(uninstallCalled, isFalse);
      expect(getModelPathCalled, isFalse);
      expect(activateFromFileCalled, isFalse);
    });

    test('activateInstalledModel when switching from another active model activates via getModelPath and fromFile without uninstalling', () async {
      final gemma2b = AppConstants.model2b;
      final qwenSpec = AppConstants.modelSpecById('qwen3-0.6b');

      var uninstallCalled = false;
      String? activatedPath;
      ModelType? activatedModelType;

      final activeQwen = InferenceModelSpec(
        name: qwenSpec.fileName,
        modelSource: FileSource('/models/${qwenSpec.fileName}'),
        modelType: qwenSpec.modelType,
        fileType: qwenSpec.fileType,
      );

      final ds = GemmaLocalDatasourceImpl(
        activeModelSpecFn: () => activeQwen,
        getModelPathFn: (name) async => '/models/$name',
        activateFromFileFn: ({
          required modelType,
          required fileType,
          required filePath,
        }) async {
          activatedModelType = modelType;
          activatedPath = filePath;
        },
        uninstallModelFn: (_) async {
          uninstallCalled = true;
        },
      );

      await ds.activateInstalledModel(
        fileName: gemma2b.fileName,
        modelType: gemma2b.modelType,
        fileType: gemma2b.fileType,
      );

      expect(uninstallCalled, isFalse);
      expect(activatedPath, '/models/${gemma2b.fileName}');
      expect(activatedModelType, gemma2b.modelType);
    });

    test('installModelFromCatalog should NEVER call uninstallModel even if activeModelSpec uses FileSource', () async {
      final spec = AppConstants.model2b;
      var uninstallCalled = false;
      var installCalled = false;

      final restoredSpec = InferenceModelSpec(
        name: spec.fileName,
        modelSource: FileSource('/models/${spec.fileName}'),
        modelType: spec.modelType,
        fileType: spec.fileType,
      );

      final ds = GemmaLocalDatasourceImpl(
        platform: TargetPlatform.android,
        activeModelSpecFn: () => restoredSpec,
        uninstallModelFn: (_) async {
          uninstallCalled = true;
        },
        installFromCatalogFn: ({
          required variant,
          required foreground,
          required onProgress,
          required cancelToken,
        }) async {
          installCalled = true;
          expect(foreground, isTrue);
          onProgress(100);
        },
      );

      var lastPercent = 0;
      await ds.installModelFromCatalog(
        variant: spec.catalogVariant,
        onProgress: (p) => lastPercent = p,
      );

      expect(installCalled, isTrue);
      expect(uninstallCalled, isFalse);
      expect(lastPercent, 100);
    });

    test('createActiveModel should pass maxConcurrentSessions=1 on Android and null on Linux', () async {
      int? androidSessions;
      final androidDs = GemmaLocalDatasourceImpl(
        platform: TargetPlatform.android,
        getActiveModelFn: ({
          required maxTokens,
          required preferredBackend,
          required supportImage,
          maxNumImages,
          activationDataType,
          maxConcurrentSessions,
        }) async {
          androidSessions = maxConcurrentSessions;
          return FakeInferenceModel();
        },
      );

      await androidDs.createActiveModel(
        preferredBackend: PreferredBackend.cpu,
        maxTokens: 2048,
      );
      expect(androidSessions, 1);

      int? linuxSessions = 99;
      final linuxDs = GemmaLocalDatasourceImpl(
        platform: TargetPlatform.linux,
        getActiveModelFn: ({
          required maxTokens,
          required preferredBackend,
          required supportImage,
          maxNumImages,
          activationDataType,
          maxConcurrentSessions,
        }) async {
          linuxSessions = maxConcurrentSessions;
          return FakeInferenceModel();
        },
      );

      await linuxDs.createActiveModel(
        preferredBackend: PreferredBackend.cpu,
        maxTokens: 2048,
      );
      expect(linuxSessions, isNull);
    });

    test('uninstallModel deletes matching native caches in supportDir without deleting conversation JSON or other models', () async {
      final tempDir = await Directory.systemTemp.createTemp('my_hero_cache_test_');
      addTearDown(() async {
        if (await tempDir.exists()) {
          await tempDir.delete(recursive: true);
        }
      });

      const targetModel = 'gemma-4-E2B-it.litertlm';
      final cacheFile1 = File('${tempDir.path}/gemma-4-E2B-it.litertlm.xnnpack_cache')
        ..writeAsStringSync('cache1');
      final cacheFile2 = File('${tempDir.path}/gemma-4-E2B-it_opencl_program_cache.bin')
        ..writeAsStringSync('cache2');
      final historyFile = File('${tempDir.path}/chat_conversations.json')
        ..writeAsStringSync('[]');
      final otherCacheFile = File('${tempDir.path}/Qwen3-0.6B.xnnpack_cache')
        ..writeAsStringSync('other');

      var uninstalledName = '';
      final ds = GemmaLocalDatasourceImpl(
        activeModelSpecFn: () => null,
        uninstallModelFn: (name) async {
          uninstalledName = name;
        },
        supportDirectoryFn: () async => tempDir,
      );

      await ds.uninstallModel(targetModel);

      expect(uninstalledName, targetModel);
      expect(await cacheFile1.exists(), isFalse);
      expect(await cacheFile2.exists(), isFalse);
      expect(await historyFile.exists(), isTrue);
      expect(await otherCacheFile.exists(), isTrue);
    });
  });
}
