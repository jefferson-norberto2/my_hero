import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';

void main() {
  group('ResponseStyle Enum', () {
    test('should return correct labels and descriptions', () {
      expect(ResponseStyle.brief.label, 'Breve');
      expect(ResponseStyle.simple.label, 'Simples');
      expect(ResponseStyle.detailed.label, 'Detalhada');

      expect(ResponseStyle.brief.description, contains('curtas'));
      expect(ResponseStyle.simple.description, contains('Linguagem clara'));
      expect(ResponseStyle.detailed.description, contains('aprofundadas'));
    });

    test('should have tailored system instructions for each style', () {
      expect(ResponseStyle.brief.systemInstruction, contains('BREVES'));
      expect(ResponseStyle.simple.systemInstruction, contains('SIMPLES'));
      expect(ResponseStyle.detailed.systemInstruction, contains('DETALHADAS'));
    });

    test('should provide correct prompt instructions concatenated to user query', () {
      expect(
        ResponseStyle.brief.promptInstruction,
        'Por favor, seja breve e direto na resposta.',
      );
      expect(
        ResponseStyle.simple.promptInstruction,
        'Por favor, me dê alguns detalhes, mas não precisa se aprofundar.',
      );
      expect(
        ResponseStyle.detailed.promptInstruction,
        'Por favor, dê mais detalhes sobre o assunto.',
      );

      expect(
        ResponseStyle.brief.applyToPrompt('Explique gravidade'),
        'Explique gravidade\n\nPor favor, seja breve e direto na resposta.',
      );
      expect(
        ResponseStyle.simple.applyToPrompt('Explique gravidade'),
        'Explique gravidade\n\nPor favor, me dê alguns detalhes, mas não precisa se aprofundar.',
      );
      expect(
        ResponseStyle.detailed.applyToPrompt('Explique gravidade'),
        'Explique gravidade\n\nPor favor, dê mais detalhes sobre o assunto.',
      );
    });

    test('fromString should parse case-insensitively with default fallback to simple', () {
      expect(ResponseStyle.fromString('brief'), ResponseStyle.brief);
      expect(ResponseStyle.fromString('breve'), ResponseStyle.brief);
      expect(ResponseStyle.fromString('detailed'), ResponseStyle.detailed);
      expect(ResponseStyle.fromString('detalhada'), ResponseStyle.detailed);
      expect(ResponseStyle.fromString('simple'), ResponseStyle.simple);
      expect(ResponseStyle.fromString('simples'), ResponseStyle.simple);
      expect(ResponseStyle.fromString(null), ResponseStyle.simple);
      expect(ResponseStyle.fromString('invalid'), ResponseStyle.simple);
    });
  });
}
