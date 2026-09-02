<script lang="ts">
	import { configData } from '$lib/free/stores/settings';
	import { configSection, fieldUpdater } from '$lib/free/stores/settingsHelpers';
	import SettingsSection from './SettingsSection.svelte';
	import ProSection from './ProSection.svelte';
	import FieldGroup from './fields/FieldGroup.svelte';
	import NumberField from './fields/NumberField.svelte';
	import ToggleField from './fields/ToggleField.svelte';

	let learning = $derived(configSection($configData, 'learning'));
	let agent = $derived(configSection($configData, 'agent'));
</script>

<SettingsSection tabId="learning">
	<!-- Free: Learning -->
	<FieldGroup label="settings.group_learning">
		<NumberField label="settings.learning.level1_min_experiences" value={Number(learning.level1_min_experiences ?? 20)} min={1} onchange={fieldUpdater('learning', 'level1_min_experiences')} />
		<NumberField label="settings.learning.level1_generations" value={Number(learning.level1_generations ?? 10)} min={1} onchange={fieldUpdater('learning', 'level1_generations')} />
		<NumberField label="settings.learning.level1_population_size" value={Number(learning.level1_population_size ?? 5)} min={1} onchange={fieldUpdater('learning', 'level1_population_size')} />
		<NumberField label="settings.learning.level2_min_failures" value={Number(learning.level2_min_failures ?? 50)} min={1} onchange={fieldUpdater('learning', 'level2_min_failures')} />
		<NumberField label="settings.learning.full_idle_minutes" value={Number(learning.full_idle_minutes ?? 10)} min={1} onchange={fieldUpdater('learning', 'full_idle_minutes')} />
		<NumberField label="settings.learning.level1_idle_minutes" value={Number(learning.level1_idle_minutes ?? 30)} min={1} onchange={fieldUpdater('learning', 'level1_idle_minutes')} />
		<NumberField label="settings.learning.level1_recheck_interval_sec" value={Number(learning.level1_recheck_interval_sec ?? 60)} min={1} onchange={fieldUpdater('learning', 'level1_recheck_interval_sec')} />
		<NumberField label="settings.learning.priority_threshold_ratio" value={Number(learning.priority_threshold_ratio ?? 0.5)} min={0} max={1} step={0.05} onchange={fieldUpdater('learning', 'priority_threshold_ratio')} />
		<NumberField label="settings.learning.active_minutes" value={Number(learning.active_minutes ?? 5)} min={1} onchange={fieldUpdater('learning', 'active_minutes')} />
		<NumberField label="settings.learning.level2_spsa_iterations" value={Number(learning.level2_spsa_iterations ?? 500)} min={1} onchange={fieldUpdater('learning', 'level2_spsa_iterations')} />
		<NumberField label="settings.learning.level2_sparse_params" value={Number(learning.level2_sparse_params ?? 200)} min={1} onchange={fieldUpdater('learning', 'level2_sparse_params')} />
		<NumberField label="settings.learning.level2_schedule_hour" value={Number(learning.level2_schedule_hour ?? 3)} min={0} max={23} onchange={fieldUpdater('learning', 'level2_schedule_hour')} />
	</FieldGroup>

	<!-- Free: Pattern Detection -->
	<FieldGroup label="settings.group_pattern_detection">
		<NumberField label="settings.learning.pattern_max_patterns" value={Number(learning.pattern_max_patterns ?? 200)} min={10} max={1000} onchange={fieldUpdater('learning', 'pattern_max_patterns')} />
		<NumberField label="settings.learning.pattern_initial_weight" value={Number(learning.pattern_initial_weight ?? 0.5)} min={0.1} max={1} step={0.05} onchange={fieldUpdater('learning', 'pattern_initial_weight')} />
		<NumberField label="settings.learning.pattern_decay_rate" value={Number(learning.pattern_decay_rate ?? 0.05)} min={0.01} max={0.5} step={0.01} onchange={fieldUpdater('learning', 'pattern_decay_rate')} />
		<NumberField label="settings.learning.pattern_boost_amount" value={Number(learning.pattern_boost_amount ?? 0.15)} min={0.01} max={0.5} step={0.01} onchange={fieldUpdater('learning', 'pattern_boost_amount')} />
		<NumberField label="settings.learning.pattern_min_weight" value={Number(learning.pattern_min_weight ?? 0.1)} min={0.01} max={0.5} step={0.01} onchange={fieldUpdater('learning', 'pattern_min_weight')} />
		<NumberField label="settings.learning.pattern_match_threshold" value={Number(learning.pattern_match_threshold ?? 0.3)} min={0.1} max={0.9} step={0.05} onchange={fieldUpdater('learning', 'pattern_match_threshold')} />
	</FieldGroup>

	<!-- Free: Agent (full width) -->
	<FieldGroup label="settings.group_agent" fullWidth>
		<ToggleField label="settings.agent.step_compaction_enabled" value={Boolean(agent.step_compaction_enabled ?? true)} onchange={fieldUpdater('agent', 'step_compaction_enabled')} />
		<NumberField label="settings.agent.step_compaction_rag_lines" value={Number(agent.step_compaction_rag_lines ?? 2)} min={1} onchange={fieldUpdater('agent', 'step_compaction_rag_lines')} />
		<NumberField label="settings.agent.step_compaction_command_head_tail" value={Number(agent.step_compaction_command_head_tail ?? 5)} min={1} onchange={fieldUpdater('agent', 'step_compaction_command_head_tail')} />
		<ToggleField label="settings.agent.reminders_enabled" value={Boolean(agent.reminders_enabled ?? true)} onchange={fieldUpdater('agent', 'reminders_enabled')} />
		<NumberField label="settings.agent.max_reminders_per_turn" value={Number(agent.max_reminders_per_turn ?? 2)} min={0} onchange={fieldUpdater('agent', 'max_reminders_per_turn')} />
		<ToggleField label="settings.agent.dangerous_command_block" value={Boolean(agent.dangerous_command_block ?? true)} onchange={fieldUpdater('agent', 'dangerous_command_block')} />
		<ToggleField label="settings.agent.tool_judge_enabled" value={Boolean(agent.tool_judge_enabled ?? true)} onchange={fieldUpdater('agent', 'tool_judge_enabled')} />
		<ToggleField label="settings.agent.meta_cognitive_enabled" value={Boolean(agent.meta_cognitive_enabled ?? true)} onchange={fieldUpdater('agent', 'meta_cognitive_enabled')} />
		<NumberField label="settings.agent.meta_cognitive_min_budget" value={Number(agent.meta_cognitive_min_budget ?? 512)} min={0} onchange={fieldUpdater('agent', 'meta_cognitive_min_budget')} />
	</FieldGroup>

	<!-- Pro: Learning extensions -->
	<ProSection>
		<FieldGroup label="settings.group_learning_pro">
			<NumberField label="settings.learning.level1_idle_minutes_pro" value={Number(learning.level1_idle_minutes_pro ?? 10)} min={1} onchange={fieldUpdater('learning', 'level1_idle_minutes_pro')} />
		</FieldGroup>
	</ProSection>
</SettingsSection>
