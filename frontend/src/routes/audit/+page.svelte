<script lang="ts">
	import type { Component } from 'svelte';
	import { t } from '$lib/i18n';
	import ProOnlyPage from '$lib/free/components/ProOnlyPage.svelte';

	// Pro 専用 (docs/p_07)。Free 版・Free バックエンドではトップへ戻す
	const loader = Object.values(
		import.meta.glob<{ default: Component }>('/src/lib/pro/components/AuditTrail.svelte')
	)[0];
</script>

<ProOnlyPage title={$t('sidebar.audit_trail')} {loader} />
