import 'package:material_ui/material_ui.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';

class ModelCatalogList extends StatefulWidget {
  final List<AiModelSpec> models;
  final String selectedModelId;
  final Set<String> installedFileNames;
  final String? loadedModelId;
  final String? deletingFileName;
  final bool enabled;
  final double maxHeight;
  final ValueChanged<AiModelSpec> onSelect;
  final Future<void> Function(AiModelSpec spec) onDelete;

  const ModelCatalogList({
    super.key,
    required this.models,
    required this.selectedModelId,
    required this.installedFileNames,
    this.loadedModelId,
    this.deletingFileName,
    this.enabled = true,
    this.maxHeight = 250,
    required this.onSelect,
    required this.onDelete,
  });

  @override
  State<ModelCatalogList> createState() => _ModelCatalogListState();
}

class _ModelCatalogListState extends State<ModelCatalogList> {
  bool _onlyInstalled = false;

  Future<void> _confirmDelete(BuildContext context, AiModelSpec spec) async {
    final isLoaded = widget.loadedModelId == spec.id;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.delete_outline, color: Colors.redAccent),
            SizedBox(width: 8),
            Expanded(child: Text('Excluir modelo?')),
          ],
        ),
        content: Text(
          'Deseja remover ${spec.displayName} (${spec.formattedSize}) do armazenamento local? '
          'Você poderá baixá-lo novamente quando quiser.'
          '${isLoaded ? '\n\nEste modelo está ativo no momento e será descarregado.' : ''}',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancelar'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(ctx).colorScheme.error,
              foregroundColor: Theme.of(ctx).colorScheme.onError,
            ),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Excluir'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await widget.onDelete(spec);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final installedCount = widget.models
        .where((m) => widget.installedFileNames.contains(m.fileName))
        .length;

    final visibleModels = _onlyInstalled
        ? widget.models
            .where((m) => widget.installedFileNames.contains(m.fileName))
            .toList()
        : widget.models;

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 4,
          alignment: WrapAlignment.center,
          children: [
            FilterChip(
              label: Text('Todos (${widget.models.length})'),
              selected: !_onlyInstalled,
              onSelected: widget.enabled
                  ? (_) => setState(() => _onlyInstalled = false)
                  : null,
            ),
            FilterChip(
              avatar: const Icon(Icons.download_done_rounded, size: 16),
              label: Text('Baixados ($installedCount)'),
              selected: _onlyInstalled,
              onSelected: widget.enabled
                  ? (_) => setState(() => _onlyInstalled = true)
                  : null,
            ),
          ],
        ),
        const SizedBox(height: 8),
        Container(
          constraints: BoxConstraints(maxHeight: widget.maxHeight),
          decoration: BoxDecoration(
            border: Border.all(
              color: theme.colorScheme.outlineVariant,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: visibleModels.isEmpty
              ? Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Text(
                    'Nenhum modelo baixado ainda.\nSelecione um modelo na aba "Todos" para baixar.',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                )
              : ClipRRect(
                  borderRadius: BorderRadius.circular(11),
                  child: ListView.separated(
                    shrinkWrap: true,
                    itemCount: visibleModels.length,
                    separatorBuilder: (_, _) => Divider(
                      height: 1,
                      color: theme.colorScheme.outlineVariant.withValues(
                        alpha: 0.5,
                      ),
                    ),
                    itemBuilder: (context, index) {
                      final spec = visibleModels[index];
                      final isSelected = spec.id == widget.selectedModelId;
                      final isInstalled =
                          widget.installedFileNames.contains(spec.fileName);
                      final isDeleting =
                          widget.deletingFileName == spec.fileName;

                      return Material(
                        color: isSelected
                            ? theme.colorScheme.primaryContainer.withValues(
                                alpha: 0.45,
                              )
                            : Colors.transparent,
                        child: InkWell(
                          onTap: widget.enabled
                              ? () => widget.onSelect(spec)
                              : null,
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 10,
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  isSelected
                                      ? Icons.radio_button_checked
                                      : Icons.radio_button_off,
                                  size: 20,
                                  color: isSelected
                                      ? Colors.teal
                                      : theme.colorScheme.outline,
                                ),
                                const SizedBox(width: 10),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        spec.displayName,
                                        style: theme.textTheme.bodyMedium
                                            ?.copyWith(
                                          fontWeight: isSelected
                                              ? FontWeight.bold
                                              : FontWeight.w500,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Wrap(
                                        spacing: 6,
                                        runSpacing: 4,
                                        crossAxisAlignment:
                                            WrapCrossAlignment.center,
                                        children: [
                                          Text(
                                            spec.formattedSize,
                                            style: theme.textTheme.labelSmall
                                                ?.copyWith(
                                              color: theme
                                                  .colorScheme.onSurfaceVariant,
                                            ),
                                          ),
                                          if (isInstalled)
                                            _CapabilityBadge(
                                              label: 'Baixado',
                                              icon: Icons.check_circle_outline,
                                              color: Colors.green.shade700,
                                              backgroundColor:
                                                  Colors.green.shade50,
                                            ),
                                          if (spec.supportsImage)
                                            _CapabilityBadge(
                                              label: 'Imagem',
                                              icon: Icons.image_outlined,
                                              color: Colors.teal.shade800,
                                              backgroundColor:
                                                  Colors.teal.shade50,
                                            ),
                                          if (spec.supportsThinking)
                                            _CapabilityBadge(
                                              label: 'Thinking',
                                              icon:
                                                  Icons.psychology_alt_outlined,
                                              color: Colors.deepPurple.shade700,
                                              backgroundColor:
                                                  Colors.deepPurple.shade50,
                                            ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                                if (isDeleting)
                                  const SizedBox(
                                    width: 24,
                                    height: 24,
                                    child: Padding(
                                      padding: EdgeInsets.all(2.0),
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    ),
                                  )
                                else if (isInstalled)
                                  IconButton(
                                    icon: const Icon(
                                      Icons.delete_outline,
                                      size: 20,
                                    ),
                                    color: theme.colorScheme.error,
                                    tooltip: 'Excluir ${spec.displayName}',
                                    onPressed: widget.enabled &&
                                            widget.deletingFileName == null
                                        ? () => _confirmDelete(context, spec)
                                        : null,
                                  ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
        ),
      ],
    );
  }
}

class _CapabilityBadge extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final Color backgroundColor;

  const _CapabilityBadge({
    required this.label,
    required this.icon,
    required this.color,
    required this.backgroundColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 3),
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
