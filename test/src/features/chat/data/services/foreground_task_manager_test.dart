import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/features/chat/data/services/foreground_task_manager.dart';

class MockForegroundTaskClient extends Mock implements ForegroundTaskClient {}

void main() {
  setUpAll(() {
    registerFallbackValue(
      AndroidNotificationOptions(
        channelId: 'fallback',
        channelName: 'fallback',
      ),
    );
    registerFallbackValue(const IOSNotificationOptions());
    registerFallbackValue(
      ForegroundTaskOptions(eventAction: ForegroundTaskEventAction.nothing()),
    );
    registerFallbackValue(<ForegroundServiceTypes>[ForegroundServiceTypes.dataSync]);
  });

  group('ForegroundTaskManager', () {
    late MockForegroundTaskClient mockClient;

    setUp(() {
      mockClient = MockForegroundTaskClient();
      when(() => mockClient.initCommunicationPort()).thenReturn(null);
      when(
        () => mockClient.init(
          androidNotificationOptions: any(named: 'androidNotificationOptions'),
          iosNotificationOptions: any(named: 'iosNotificationOptions'),
          foregroundTaskOptions: any(named: 'foregroundTaskOptions'),
        ),
      ).thenReturn(null);
      when(() => mockClient.checkNotificationPermission())
          .thenAnswer((_) async => NotificationPermission.granted);
      when(() => mockClient.requestNotificationPermission())
          .thenAnswer((_) async => NotificationPermission.granted);
      when(
        () => mockClient.startService(
          serviceId: any(named: 'serviceId'),
          serviceTypes: any(named: 'serviceTypes'),
          notificationTitle: any(named: 'notificationTitle'),
          notificationText: any(named: 'notificationText'),
          callback: any(named: 'callback'),
        ),
      ).thenAnswer((_) async => const ServiceRequestSuccess());
      when(
        () => mockClient.updateService(
          notificationTitle: any(named: 'notificationTitle'),
          notificationText: any(named: 'notificationText'),
        ),
      ).thenAnswer((_) async => const ServiceRequestSuccess());
      when(() => mockClient.stopService())
          .thenAnswer((_) async => const ServiceRequestSuccess());
    });

    test('performs no native calls when running outside Android (e.g. Linux desktop)', () async {
      final manager = ForegroundTaskManager(
        client: mockClient,
        isAndroid: () => false,
      );

      await manager.initialize();
      await manager.beginChatGeneration();
      await manager.beginModelDownload('llama3.2:3b');
      await manager.updateModelDownloadProgress('llama3.2:3b', percent: 50);
      await manager.endModelDownload();
      await manager.endChatGeneration();

      verifyZeroInteractions(mockClient);
    });

    test('starts dataSync foreground service on Android for chat and stops when finished', () async {
      var running = false;
      when(() => mockClient.isRunningService).thenAnswer((_) async => running);
      when(
        () => mockClient.startService(
          serviceId: any(named: 'serviceId'),
          serviceTypes: any(named: 'serviceTypes'),
          notificationTitle: any(named: 'notificationTitle'),
          notificationText: any(named: 'notificationText'),
          callback: any(named: 'callback'),
        ),
      ).thenAnswer((_) async {
        running = true;
        return const ServiceRequestSuccess();
      });
      when(() => mockClient.stopService()).thenAnswer((_) async {
        running = false;
        return const ServiceRequestSuccess();
      });

      final manager = ForegroundTaskManager(
        client: mockClient,
        isAndroid: () => true,
      );

      await manager.beginChatGeneration();

      verify(
        () => mockClient.startService(
          serviceId: ForegroundTaskManager.foregroundServiceId,
          serviceTypes: const [ForegroundServiceTypes.dataSync],
          notificationTitle: 'My Hero',
          notificationText: 'Gerando resposta...',
          callback: startForegroundTaskCallback,
        ),
      ).called(1);

      await manager.endChatGeneration();
      verify(() => mockClient.stopService()).called(1);
    });

    test('updates download notification progress and skips redundant identical updates', () async {
      var running = false;
      when(() => mockClient.isRunningService).thenAnswer((_) async => running);
      when(
        () => mockClient.startService(
          serviceId: any(named: 'serviceId'),
          serviceTypes: any(named: 'serviceTypes'),
          notificationTitle: any(named: 'notificationTitle'),
          notificationText: any(named: 'notificationText'),
          callback: any(named: 'callback'),
        ),
      ).thenAnswer((_) async {
        running = true;
        return const ServiceRequestSuccess();
      });
      when(() => mockClient.stopService()).thenAnswer((_) async {
        running = false;
        return const ServiceRequestSuccess();
      });

      final manager = ForegroundTaskManager(
        client: mockClient,
        isAndroid: () => true,
      );

      await manager.beginModelDownload('qwen2.5:7b');
      await manager.updateModelDownloadProgress('qwen2.5:7b', percent: 35);
      await manager.updateModelDownloadProgress('qwen2.5:7b', percent: 35);
      await manager.updateModelDownloadProgress('qwen2.5:7b', percent: 80);
      await manager.endModelDownload();

      verify(
        () => mockClient.startService(
          serviceId: ForegroundTaskManager.foregroundServiceId,
          serviceTypes: const [ForegroundServiceTypes.dataSync],
          notificationTitle: 'My Hero',
          notificationText: 'Baixando modelo qwen2.5:7b...',
          callback: startForegroundTaskCallback,
        ),
      ).called(1);
      verify(
        () => mockClient.updateService(
          notificationTitle: 'My Hero',
          notificationText: 'Baixando modelo qwen2.5:7b: 35%',
        ),
      ).called(1);
      verify(
        () => mockClient.updateService(
          notificationTitle: 'My Hero',
          notificationText: 'Baixando modelo qwen2.5:7b: 80%',
        ),
      ).called(1);
      verify(() => mockClient.stopService()).called(1);
    });

    test('keeps service active when chat ends while a model download is still running', () async {
      var running = false;
      when(() => mockClient.isRunningService).thenAnswer((_) async => running);
      when(
        () => mockClient.startService(
          serviceId: any(named: 'serviceId'),
          serviceTypes: any(named: 'serviceTypes'),
          notificationTitle: any(named: 'notificationTitle'),
          notificationText: any(named: 'notificationText'),
          callback: any(named: 'callback'),
        ),
      ).thenAnswer((_) async {
        running = true;
        return const ServiceRequestSuccess();
      });
      when(() => mockClient.stopService()).thenAnswer((_) async {
        running = false;
        return const ServiceRequestSuccess();
      });

      final manager = ForegroundTaskManager(
        client: mockClient,
        isAndroid: () => true,
      );

      await manager.beginModelDownload('llama3.2:3b');
      await manager.updateModelDownloadProgress('llama3.2:3b', percent: 40);
      await manager.beginChatGeneration();
      await manager.endChatGeneration();

      verifyNever(() => mockClient.stopService());
      verify(
        () => mockClient.updateService(
          notificationTitle: 'My Hero',
          notificationText: 'Baixando modelo llama3.2:3b: 40%',
        ),
      ).called(2);

      await manager.endModelDownload();
      verify(() => mockClient.stopService()).called(1);
    });

    test('requests notification permission once on Android when denied initially', () async {
      when(() => mockClient.isRunningService).thenAnswer((_) async => false);
      when(() => mockClient.checkNotificationPermission())
          .thenAnswer((_) async => NotificationPermission.denied);

      final manager = ForegroundTaskManager(
        client: mockClient,
        isAndroid: () => true,
      );

      await manager.beginChatGeneration();
      await manager.beginModelDownload('llama3.2:3b');

      verify(() => mockClient.checkNotificationPermission()).called(1);
      verify(() => mockClient.requestNotificationPermission()).called(1);
    });
  });
}
