import 'package:flutter/material.dart';
import 'package:my_hero/src/shared/models/response_style.dart';

/// Card section displaying the segmented response style selector and description.
class ResponseStyleSection extends StatelessWidget {
  final ResponseStyle selectedStyle;
  final ValueChanged<ResponseStyle> onStyleChanged;

  const ResponseStyleSection({
    super.key,
    required this.selectedStyle,
    required this.onStyleChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.4),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: theme.colorScheme.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Row(
              children: [
                Icon(Icons.auto_awesome_outlined, color: Colors.teal),
                SizedBox(width: 8),
                Text(
                  'Estilo de Resposta',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SegmentedButton<ResponseStyle>(
              segments: const [
                ButtonSegment<ResponseStyle>(
                  value: ResponseStyle.brief,
                  label: Text('Curta'),
                  icon: Icon(Icons.bolt),
                ),
                ButtonSegment<ResponseStyle>(
                  value: ResponseStyle.simple,
                  label: Text('Padrão'),
                  icon: Icon(Icons.lightbulb_outline),
                ),
                ButtonSegment<ResponseStyle>(
                  value: ResponseStyle.detailed,
                  label: Text('Longa'),
                  icon: Icon(Icons.menu_book),
                ),
              ],
              selected: {selectedStyle},
              onSelectionChanged: (newSelection) {
                onStyleChanged(newSelection.first);
              },
            ),
            const SizedBox(height: 10),
            Text(
              selectedStyle.description,
              style: TextStyle(
                fontSize: 12,
                color: Colors.teal.shade800,
                fontStyle: FontStyle.italic,
              ),
            ),
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.teal.shade50,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.teal.shade200),
              ),
              child: const Row(
                children: [
                  Icon(Icons.sync, color: Colors.teal, size: 22),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'O contexto desta conversa e o novo modelo serão mantidos para as próximas perguntas.',
                      style: TextStyle(fontSize: 12, color: Colors.black87),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
