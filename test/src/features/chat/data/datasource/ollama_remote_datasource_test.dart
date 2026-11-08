import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';

class MockDio extends Mock implements Dio {}

void main() {
  late MockDio mockDio;
  late DioClient dioClient;
  late OllamaRemoteDatasourceImpl datasource;

  setUp(() {
    mockDio = MockDio();
    dioClient = DioClient(mockDio);
    datasource = OllamaRemoteDatasourceImpl(dioClient);
  });

  group('OllamaRemoteDatasourceImpl', () {
    test('checkHealth returns true on 200 status', () async {
      when(
        () => mockDio.get(
          any(),
          options: any(named: 'options'),
        ),
      ).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      final result = await datasource.checkHealth();
      expect(result, isTrue);
    });

    test('checkHealth returns false on network exception', () async {
      when(
        () => mockDio.get(
          any(),
          options: any(named: 'options'),
        ),
      ).thenThrow(DioException(requestOptions: RequestOptions(path: '')));

      final result = await datasource.checkHealth();
      expect(result, isFalse);
    });

    test('getAvailableModels parses models and identifies loaded ones', () async {
      // Mock /api/ps (loaded models)
      when(
        () => mockDio.get(
          any(that: contains('/api/ps')),
          options: any(named: 'options'),
        ),
      ).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
          data: {
            'models': [
              {'name': 'llama3.2:3b'},
            ],
          },
        ),
      );

      // Mock /api/tags (all models)
      when(
        () => mockDio.get(
          any(that: contains('/api/tags')),
          options: any(named: 'options'),
        ),
      ).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
          data: {
            'models': [
              {
                'name': 'llama3.2:3b',
                'model': 'llama3.2:3b',
                'size': 2000000000,
                'details': {
                  'parameter_size': '3.2B',
                  'family': 'llama',
                },
              },
              {
                'name': 'deepseek-r1:8b',
                'model': 'deepseek-r1:8b',
                'size': 5000000000,
                'details': {
                  'parameter_size': '8.2B',
                  'family': 'qwen',
                },
              },
            ],
          },
        ),
      );

      final models = await datasource.getAvailableModels();

      expect(models.length, 2);
      expect(models[0].name, 'llama3.2:3b');
      expect(models[0].isLoaded, isTrue);
      expect(models[1].name, 'deepseek-r1:8b');
      expect(models[1].isLoaded, isFalse);
    });

    test('preloadModel calls POST /api/generate with model name', () async {
      when(
        () => mockDio.post(
          any(that: contains('/api/generate')),
          data: any(named: 'data'),
          options: any(named: 'options'),
        ),
      ).thenAnswer(
        (_) async => Response(
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      await datasource.preloadModel('llama3.2:3b');

      verify(
        () => mockDio.post(
          any(that: contains('/api/generate')),
          data: {'model': 'llama3.2:3b'},
          options: any(named: 'options'),
        ),
      ).called(1);
    });

    test('generateChatResponseAsync yields content chunks', () async {
      final jsonLines = [
        '{"message": {"role": "assistant", "content": "Olá"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": " mundo"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "!"}, "done": true}\n',
      ];

      final byteStream = Stream.fromIterable(
        jsonLines.map((l) => Uint8List.fromList(utf8.encode(l))),
      );

      final responseBody = ResponseBody(
        byteStream,
        200,
        headers: {
          Headers.contentTypeHeader: [Headers.jsonContentType],
        },
      );

      when(
        () => mockDio.post<ResponseBody>(
          any(that: contains('/api/chat')),
          data: any(named: 'data'),
          options: any(named: 'options'),
          cancelToken: any(named: 'cancelToken'),
        ),
      ).thenAnswer(
        (_) async => Response<ResponseBody>(
          data: responseBody,
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      final stream = datasource.generateChatResponseAsync(
        modelName: 'llama3.2:3b',
        messages: [
          {'role': 'user', 'content': 'Oi'},
        ],
      );

      final chunks = await stream.toList();

      expect(chunks, ['Olá', ' mundo', '!']);
    });
  });
}
