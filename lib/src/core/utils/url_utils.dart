import '../constants/app_constants.dart';

/// Utilitário para sanitizar e normalizar URLs e endereços de IP do Ollama.
///
/// Exemplos:
/// - `192.168.1.50` -> `http://192.168.1.50:11434`
/// - `192.168.1.50:11434` -> `http://192.168.1.50:11434`
/// - `http://192.168.1.50:8000/` -> `http://192.168.1.50:8000`
/// - `localhost` -> `http://localhost:11434`
/// - `""` -> `http://localhost:11434`
String normalizeOllamaUrl(String input) {
  var raw = input.trim();
  if (raw.isEmpty) {
    return AppConstants.ollamaBaseUrl;
  }

  // Remove barras finais
  while (raw.endsWith('/')) {
    raw = raw.substring(0, raw.length - 1);
  }

  // Se não foi informado protocolo http ou https, adiciona http://
  if (!raw.startsWith('http://') && !raw.startsWith('https://')) {
    raw = 'http://$raw';
  }

  final uri = Uri.tryParse(raw);
  if (uri == null || uri.host.isEmpty) {
    return raw;
  }

  // Se nenhuma porta foi informada, define a padrão do Ollama (11434)
  final port = uri.hasPort ? uri.port : 11434;
  final path = uri.path;

  return '${uri.scheme}://${uri.host}:$port$path';
}
