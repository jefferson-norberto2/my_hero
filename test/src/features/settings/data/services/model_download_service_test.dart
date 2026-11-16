import 'dart:async';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/features/chat/data/services/foreground_task_manager.dart';
import 'package:my_hero/src/features/settings/data/services/model_download_service.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';

class MockOllamaRemoteDatasource extends Mock
    implements OllamaRemoteDatasource {}

class MockForegroundTaskManager extends Mock implements ForegroundTaskManager {}

void main() {
  late MockOllamaRemoteDatasource mockDatasource;
  late MockForegroundTaskManager mockForegroundTaskManager;
  late ModelDownloadService service;

  setUp(() {
    mockDatasource = MockOllamaRemoteDatasource();
    mockForegroundTaskManager = MockForegroundTaskManager();

    when(() => mockForegroundTaskManager.beginModelDownload(any()))
        .thenAnswer((_) async {});
    when(
      () => mockForegroundTaskManager.updateModelDownloadProgress(
        any(),
        percent: any(named: 'percent'),
        statusText: any(named: 'statusText'),
      ),
    ).thenAnswer((_) async {});
    when(() => mockForegroundTaskManager.endModelDownload())
        .thenAnswer((_) async {});
    when(() => mockDatasource.cancelModelPull()).thenReturn(null);

    service = ModelDownloadService(
      ollamaDatasource: mockDatasource,
      foregroundTaskManager: mockForegroundTaskManager,
      retryDelay: Duration.zero,
    );
  });

  tearDown(() {
    service.dispose();
  });

  group('ModelDownloadService', () {
    test('downloads model, updates foreground progress, and stops service on completion', () async {
      when(() => mockDatasource.pullModel('llama3.2:3b')).thenAnswer(
        (_) => Stream.fromIterable(const [
          ModelDownloadState(
            modelName: 'llama3.2:3b',
            status: ModelDownloadStatus.downloading,
            statusMessage: 'downloading layer',
            completedBytes: 35,
            totalBytes: 100,
          ),
          ModelDownloadState(
            modelName: 'llama3.2:3b',
            status: ModelDownloadStatus.completed,
            statusMessage: 'Modelo "llama3.2:3b" baixado com sucesso!',
            completedBytes: 100,
            totalBytes: 100,
          ),
        ]),
      );

      final emitted = <ModelDownloadState>[];
      final sub = service.stateStream.listen(emitted.add);

      final result = await service.downloadModel('llama3.2:3b');
      await Future<void>.delayed(Duration.zero);
      await sub.cancel();

      expect(result.status, ModelDownloadStatus.completed);
      expect(service.currentState.status, ModelDownloadStatus.completed);
      expect(emitted.any((s) => s.progressPercent == 35), isTrue);

      verify(() => mockForegroundTaskManager.beginModelDownload('llama3.2:3b'))
          .called(1);
      verify(
        () => mockForegroundTaskManager.updateModelDownloadProgress(
          'llama3.2:3b',
          percent: 35,
        ),
      ).called(1);
      verify(() => mockForegroundTaskManager.endModelDownload()).called(1);
    });

    test('retries POST /api/pull up to 3 times on unexpected connection drop and resumes progress', () async {
      var attemptCount = 0;
      when(() => mockDatasource.pullModel('qwen2.5:7b')).thenAnswer((_) {
        attemptCount++;
        if (attemptCount <= 2) {
          return Stream<ModelDownloadState>.multi((controller) {
            controller.add(
              const ModelDownloadState(
                modelName: 'qwen2.5:7b',
                status: ModelDownloadStatus.downloading,
                statusMessage: 'downloading',
                completedBytes: 50,
                totalBytes: 100,
              ),
            );
            controller.addError(
              DioException(
                requestOptions: RequestOptions(path: '/api/pull'),
                type: DioExceptionType.connectionError,
                message: 'Socket dropped',
              ),
            );
            controller.close();
          });
        }
        return Stream.fromIterable(const [
          ModelDownloadState(
            modelName: 'qwen2.5:7b',
            status: ModelDownloadStatus.downloading,
            statusMessage: 'downloading',
            completedBytes: 90,
            totalBytes: 100,
          ),
          ModelDownloadState(
            modelName: 'qwen2.5:7b',
            status: ModelDownloadStatus.completed,
            statusMessage: 'Modelo "qwen2.5:7b" baixado com sucesso!',
            completedBytes: 100,
            totalBytes: 100,
          ),
        ]);
      });

      final emitted = <ModelDownloadState>[];
      final sub = service.stateStream.listen(emitted.add);

      final result = await service.downloadModel('qwen2.5:7b');
      await Future<void>.delayed(Duration.zero);
      await sub.cancel();

      expect(attemptCount, 3);
      expect(result.status, ModelDownloadStatus.completed);
      expect(
        emitted.where((s) => s.status == ModelDownloadStatus.reconnecting).length,
        2,
      );
      verify(() => mockDatasource.pullModel('qwen2.5:7b')).called(3);
      verify(() => mockForegroundTaskManager.endModelDownload()).called(1);
    });

    test('stops with error after 3 reconnection attempts when connection keeps dropping', () async {
      when(() => mockDatasource.pullModel('deepseek-r1:8b')).thenAnswer(
        (_) => Stream<ModelDownloadState>.error(
          const SocketException('Network unreachable'),
        ),
      );

      final result = await service.downloadModel('deepseek-r1:8b');

      expect(result.status, ModelDownloadStatus.error);
      expect(result.retryAttempt, ModelDownloadService.maxReconnectAttempts);
      verify(() => mockDatasource.pullModel('deepseek-r1:8b')).called(4);
      verify(() => mockForegroundTaskManager.endModelDownload()).called(1);
    });

    test('retries when stream terminates prematurely without success chunk', () async {
      var calls = 0;
      when(() => mockDatasource.pullModel('gemma3:4b')).thenAnswer((_) {
        calls++;
        if (calls == 1) {
          return Stream.fromIterable(const [
            ModelDownloadState(
              modelName: 'gemma3:4b',
              status: ModelDownloadStatus.downloading,
              statusMessage: 'pulling manifest',
            ),
          ]);
        }
        return Stream.fromIterable(const [
          ModelDownloadState(
            modelName: 'gemma3:4b',
            status: ModelDownloadStatus.completed,
            statusMessage: 'Modelo "gemma3:4b" baixado com sucesso!',
            completedBytes: 100,
            totalBytes: 100,
          ),
        ]);
      });

      final result = await service.downloadModel('gemma3:4b');

      expect(calls, 2);
      expect(result.status, ModelDownloadStatus.completed);
    });

    test('does not retry on explicit OllamaPullServerException and ends foreground service', () async {
      when(() => mockDatasource.pullModel('unknown:model')).thenAnswer(
        (_) => Stream<ModelDownloadState>.error(
          const OllamaPullServerException('pull model manifest: file does not exist'),
        ),
      );

      final result = await service.downloadModel('unknown:model');

      expect(result.status, ModelDownloadStatus.error);
      expect(result.errorMessage, contains('file does not exist'));
      verify(() => mockDatasource.pullModel('unknown:model')).called(1);
      verify(() => mockForegroundTaskManager.endModelDownload()).called(1);
    });

    test('cancelDownload aborts active retry delay immediately and stops foreground service', () async {
      final slowRetryService = ModelDownloadService(
        ollamaDatasource: mockDatasource,
        foregroundTaskManager: mockForegroundTaskManager,
        retryDelay: const Duration(seconds: 30),
      );

      when(() => mockDatasource.pullModel('llama3.2:3b')).thenAnswer(
        (_) => Stream<ModelDownloadState>.error(
          const SocketException('Dropped'),
        ),
      );

      final downloadFuture = slowRetryService.downloadModel('llama3.2:3b');
      await Future<void>.delayed(const Duration(milliseconds: 20));
      await slowRetryService.cancelDownload();
      final result = await downloadFuture;

      expect(result.status, ModelDownloadStatus.cancelled);
      verify(() => mockDatasource.pullModel('llama3.2:3b')).called(1);
      verify(() => mockForegroundTaskManager.endModelDownload()).called(1);
      slowRetryService.dispose();
    });
  });
}
