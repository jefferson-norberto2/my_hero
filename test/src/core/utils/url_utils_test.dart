import 'package:flutter_test/flutter_test.dart';
import 'package:my_hero/src/core/utils/url_utils.dart';

void main() {
  group('normalizeOllamaUrl', () {
    test('returns default localhost:11434 when input is empty or whitespace', () {
      expect(normalizeOllamaUrl(''), equals('http://localhost:11434'));
      expect(normalizeOllamaUrl('   '), equals('http://localhost:11434'));
    });

    test('normalizes plain IP to http://IP:11434', () {
      expect(normalizeOllamaUrl('192.168.1.50'), equals('http://192.168.1.50:11434'));
      expect(normalizeOllamaUrl('10.0.0.12'), equals('http://10.0.0.12:11434'));
    });

    test('normalizes IP with port correctly', () {
      expect(normalizeOllamaUrl('192.168.1.50:11434'), equals('http://192.168.1.50:11434'));
      expect(normalizeOllamaUrl('192.168.1.50:8080'), equals('http://192.168.1.50:8080'));
    });

    test('handles input with http/https scheme and trailing slash', () {
      expect(normalizeOllamaUrl('http://192.168.1.50:11434/'), equals('http://192.168.1.50:11434'));
      expect(normalizeOllamaUrl('http://192.168.1.50/'), equals('http://192.168.1.50:11434'));
      expect(normalizeOllamaUrl('https://ollama.lan:11434/'), equals('https://ollama.lan:11434'));
    });

    test('handles localhost and 127.0.0.1', () {
      expect(normalizeOllamaUrl('localhost'), equals('http://localhost:11434'));
      expect(normalizeOllamaUrl('localhost:11434'), equals('http://localhost:11434'));
      expect(normalizeOllamaUrl('127.0.0.1'), equals('http://127.0.0.1:11434'));
      expect(normalizeOllamaUrl('127.0.0.1:11434'), equals('http://127.0.0.1:11434'));
    });
  });
}
