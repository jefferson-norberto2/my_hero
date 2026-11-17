import 'package:flutter/material.dart';

/// Contextual AppBar shown while the user is selecting chat messages to export.
class ChatSelectionAppBar extends StatelessWidget
    implements PreferredSizeWidget {
  final int selectedCount;
  final int selectableCount;
  final bool allSelected;
  final bool isExporting;
  final bool isCompact;
  final VoidCallback onClose;
  final VoidCallback onToggleSelectAll;
  final VoidCallback onExport;

  static const double _compactBottomHeight = 52.0;

  const ChatSelectionAppBar({
    super.key,
    required this.selectedCount,
    required this.selectableCount,
    required this.allSelected,
    required this.isExporting,
    required this.isCompact,
    required this.onClose,
    required this.onToggleSelectAll,
    required this.onExport,
  });

  @override
  Size get preferredSize => Size.fromHeight(
        kToolbarHeight + (isCompact ? _compactBottomHeight : 0.0),
      );

  String get _selectionCounterLabel =>
      selectedCount == 1 ? '1 selecionada' : '$selectedCount selecionadas';

  String get _selectAllLabel =>
      allSelected ? 'Desmarcar tudo' : 'Selecionar tudo';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final canExport = selectedCount > 0 && !isExporting;

    return AppBar(
      backgroundColor: theme.colorScheme.secondaryContainer,
      foregroundColor: theme.colorScheme.onSecondaryContainer,
      leading: IconButton(
        icon: const Icon(Icons.close),
        tooltip: 'Fechar seleção',
        onPressed: isExporting ? null : onClose,
      ),
      titleSpacing: 0,
      title: Text(
        _selectionCounterLabel,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
        overflow: TextOverflow.ellipsis,
      ),
      actions: [
        TextButton(
          onPressed: isExporting || selectableCount == 0
              ? null
              : onToggleSelectAll,
          child: Text(_selectAllLabel),
        ),
        if (!isCompact)
          Padding(
            padding: const EdgeInsets.only(right: 12.0, left: 4.0),
            child: FilledButton.icon(
              onPressed: canExport ? onExport : null,
              icon: const Icon(Icons.file_download_outlined, size: 18),
              label: const Text('Exportar'),
            ),
          ),
      ],
      bottom: isCompact
          ? PreferredSize(
              preferredSize: const Size.fromHeight(_compactBottomHeight),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16.0, 0.0, 16.0, 8.0),
                child: SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: canExport ? onExport : null,
                    icon: const Icon(Icons.file_download_outlined, size: 18),
                    label: const Text('Exportar'),
                  ),
                ),
              ),
            )
          : null,
    );
  }
}
