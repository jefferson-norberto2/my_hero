import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_hero/src/core/network/dio_client.dart';
import 'package:my_hero/src/features/chat/data/datasource/ollama_remote_datasource.dart';
import 'package:my_hero/src/shared/models/chat_stream_chunk.dart';

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

    test('getAvailableModels parses models, identifies loaded ones and extracts vram/contextLength', () async {
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
              {
                'name': 'llama3.2:3b',
                'size_vram': 2500000000,
                'context_length': 8192,
              },
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
      expect(models[0].sizeVram, 2500000000);
      expect(models[0].contextLength, 8192);

      expect(models[1].name, 'deepseek-r1:8b');
      expect(models[1].isLoaded, isFalse);
      expect(models[1].sizeVram, 0);
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

    test('generateChatResponseAsync yields content chunks and calls onMetrics when done', () async {
      final jsonLines = [
        '{"message": {"role": "assistant", "content": "Olá"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": " mundo"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "!"}, "done": true, "prompt_eval_count": 42, "eval_count": 3}\n',
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

      int? capturedPromptTokens;
      int? capturedEvalTokens;

      final stream = datasource.generateChatResponseAsync(
        modelName: 'llama3.2:3b',
        messages: [
          {'role': 'user', 'content': 'Oi'},
        ],
        onMetrics: (p, e) {
          capturedPromptTokens = p;
          capturedEvalTokens = e;
        },
      );

      final chunks = await stream.toList();

      expect(
        chunks,
        const [
          ChatStreamChunk(contentDelta: 'Olá'),
          ChatStreamChunk(contentDelta: ' mundo'),
          ChatStreamChunk(contentDelta: '!', isDone: true),
        ],
      );
      expect(capturedPromptTokens, 42);
      expect(capturedEvalTokens, 3);
    });

    test('generateChatResponseAsync separates native message.thinking and message.content', () async {
      final jsonLines = [
        '{"message": {"role": "assistant", "thinking": "Analisando"}, "done": false}\n',
        '{"message": {"role": "assistant", "thinking": " passo 1"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "Resposta final"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "!"}, "done": true}\n',
      ];

      final byteStream = Stream.fromIterable(
        jsonLines.map((l) => Uint8List.fromList(utf8.encode(l))),
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
          data: ResponseBody(byteStream, 200),
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      final chunks = await datasource.generateChatResponseAsync(
        modelName: 'qwen3:8b',
        messages: [
          {'role': 'user', 'content': 'Oi'},
        ],
      ).toList();

      expect(
        chunks,
        const [
          ChatStreamChunk(thinkingDelta: 'Analisando', isThinking: true),
          ChatStreamChunk(thinkingDelta: ' passo 1', isThinking: true),
          ChatStreamChunk(contentDelta: 'Resposta final', isThinking: false),
          ChatStreamChunk(contentDelta: '!', isThinking: false, isDone: true),
        ],
      );
    });

    test('generateChatResponseAsync extracts split <think> tags from content', () async {
      final jsonLines = [
        '{"message": {"role": "assistant", "content": "<thi"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "nk>Pensando"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": " muito</th"}, "done": false}\n',
        '{"message": {"role": "assistant", "content": "ink>Oi!"}, "done": true}\n',
      ];

      final byteStream = Stream.fromIterable(
        jsonLines.map((l) => Uint8List.fromList(utf8.encode(l))),
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
          data: ResponseBody(byteStream, 200),
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      final chunks = await datasource.generateChatResponseAsync(
        modelName: 'deepseek-r1:8b',
        messages: [
          {'role': 'user', 'content': 'Oi'},
        ],
      ).toList();

      expect(chunks.map((c) => c.thinkingDelta).join(), 'Pensando muito');
      expect(chunks.map((c) => c.contentDelta).join(), 'Oi!');
      expect(chunks.last.isThinking, isFalse);
      expect(chunks.last.isDone, isTrue);
    });

    test('generateChatResponseAsync handles top-level thinking and response fields', () async {
      final jsonLines = [
        '{"thinking": "Raciocínio", "response": "", "done": false}\n',
        '{"thinking": "", "response": "Pronto", "done": true}\n',
      ];

      final byteStream = Stream.fromIterable(
        jsonLines.map((l) => Uint8List.fromList(utf8.encode(l))),
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
          data: ResponseBody(byteStream, 200),
          requestOptions: RequestOptions(path: ''),
          statusCode: 200,
        ),
      );

      final chunks = await datasource.generateChatResponseAsync(
        modelName: 'qwen3:8b',
        messages: [
          {'role': 'user', 'content': 'Oi'},
        ],
      ).toList();

      expect(
        chunks,
        const [
          ChatStreamChunk(thinkingDelta: 'Raciocínio', isThinking: true),
          ChatStreamChunk(contentDelta: 'Pronto', isThinking: false, isDone: true),
        ],
      );
    });
  });
}
