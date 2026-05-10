-- Script SQL de vérification du système de workflows
-- Usage: psql -U votre_user -d votre_database -f check_workflows.sql

\echo '========================================='
\echo 'VÉRIFICATION DU SYSTÈME DE WORKFLOWS'
\echo '========================================='
\echo ''

-- 1. Vérifier l'existence des tables
\echo '1. Tables existantes:'
\echo '---------------------'
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('generated_workflows', 'workflow_actions', 'workflow_executions')
ORDER BY table_name;
\echo ''

-- 2. Compter les workflows
\echo '2. Nombre de workflows:'
\echo '-----------------------'
SELECT 
    COUNT(*) as total_workflows,
    SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_workflows,
    SUM(CASE WHEN NOT is_active THEN 1 ELSE 0 END) as inactive_workflows
FROM generated_workflows;
\echo ''

-- 3. Lister les workflows récents
\echo '3. Workflows récents (5 derniers):'
\echo '-----------------------------------'
SELECT 
    id,
    prompt_normalized,
    is_active,
    execution_count,
    avg_execution_time_ms,
    created_at
FROM generated_workflows
ORDER BY created_at DESC
LIMIT 5;
\echo ''

-- 4. Statistiques des actions
\echo '4. Statistiques des actions:'
\echo '----------------------------'
SELECT 
    action_type,
    COUNT(*) as count
FROM workflow_actions
GROUP BY action_type
ORDER BY count DESC;
\echo ''

-- 5. Statistiques des exécutions
\echo '5. Statistiques des exécutions:'
\echo '-------------------------------'
SELECT 
    COUNT(*) as total_executions,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_executions,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_executions,
    AVG(execution_time_ms) as avg_execution_time_ms,
    MIN(execution_time_ms) as min_execution_time_ms,
    MAX(execution_time_ms) as max_execution_time_ms
FROM workflow_executions;
\echo ''

-- 6. Workflows les plus utilisés
\echo '6. Workflows les plus utilisés (Top 5):'
\echo '----------------------------------------'
SELECT 
    gw.id,
    gw.prompt_normalized,
    gw.execution_count,
    gw.avg_execution_time_ms,
    COUNT(we.id) as total_runs
FROM generated_workflows gw
LEFT JOIN workflow_executions we ON gw.id = we.workflow_id
GROUP BY gw.id, gw.prompt_normalized, gw.execution_count, gw.avg_execution_time_ms
ORDER BY gw.execution_count DESC
LIMIT 5;
\echo ''

-- 7. Dernières exécutions
\echo '7. Dernières exécutions (5 dernières):'
\echo '---------------------------------------'
SELECT 
    we.id,
    gw.prompt_normalized,
    we.success,
    we.execution_time_ms,
    we.error_message,
    we.created_at
FROM workflow_executions we
JOIN generated_workflows gw ON we.workflow_id = gw.id
ORDER BY we.created_at DESC
LIMIT 5;
\echo ''

-- 8. Workflows avec erreurs
\echo '8. Workflows avec erreurs récentes:'
\echo '-----------------------------------'
SELECT 
    gw.id,
    gw.prompt_normalized,
    COUNT(we.id) as error_count,
    MAX(we.created_at) as last_error_at
FROM generated_workflows gw
JOIN workflow_executions we ON gw.id = we.workflow_id
WHERE we.success = false
GROUP BY gw.id, gw.prompt_normalized
ORDER BY error_count DESC
LIMIT 5;
\echo ''

-- 9. Performance moyenne par workflow
\echo '9. Performance moyenne par workflow:'
\echo '------------------------------------'
SELECT 
    gw.id,
    gw.prompt_normalized,
    gw.execution_count,
    gw.avg_execution_time_ms as stored_avg_ms,
    AVG(we.execution_time_ms) as calculated_avg_ms,
    MIN(we.execution_time_ms) as min_ms,
    MAX(we.execution_time_ms) as max_ms
FROM generated_workflows gw
LEFT JOIN workflow_executions we ON gw.id = we.workflow_id
WHERE gw.execution_count > 0
GROUP BY gw.id, gw.prompt_normalized, gw.execution_count, gw.avg_execution_time_ms
ORDER BY gw.execution_count DESC
LIMIT 10;
\echo ''

-- 10. Vérifier les screenshots
\echo '10. Statistiques des screenshots:'
\echo '---------------------------------'
SELECT 
    COUNT(*) as total_screenshots,
    COUNT(DISTINCT session_id) as sessions_with_screenshots,
    AVG(LENGTH(content)) as avg_screenshot_size_bytes
FROM messages
WHERE message_type = 'screenshot';
\echo ''

-- 11. Sessions récentes avec screenshots
\echo '11. Sessions récentes avec screenshots:'
\echo '---------------------------------------'
SELECT 
    s.id,
    s.topic,
    COUNT(m.id) as screenshot_count,
    MAX(m.created_at) as last_screenshot_at
FROM sessions s
JOIN messages m ON s.id = m.session_id
WHERE m.message_type = 'screenshot'
GROUP BY s.id, s.topic
ORDER BY last_screenshot_at DESC
LIMIT 5;
\echo ''

\echo '========================================='
\echo 'FIN DE LA VÉRIFICATION'
\echo '========================================='
