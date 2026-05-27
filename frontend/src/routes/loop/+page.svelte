<script lang="ts">
	import LoopControlBar from '$lib/free/components/LoopControlBar.svelte';
	import LoopTaskPanel from '$lib/free/components/LoopTaskPanel.svelte';
	import LoopEventConsole from '$lib/free/components/LoopEventConsole.svelte';
</script>

<div class="loop-page">
	<LoopControlBar />
	<LoopTaskPanel />
	<div class="console-wrap">
		<LoopEventConsole />
	</div>
</div>

<style>
	.loop-page {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.console-wrap {
		flex: 1;
		min-height: 0;
	}
</style>
