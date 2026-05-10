#!/usr/bin/env python3
"""
Script de test automatisé pour le système de workflows
Usage: python test_workflows.py
"""

import requests
import time
import json
from typing import Dict, List

BASE_URL = "http://127.0.0.1:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def test_backend_health() -> bool:
    """Test 1: Vérifier que le backend est accessible"""
    print_info("Test 1: Vérification de la santé du backend...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Backend accessible")
            return True
        else:
            print_error(f"Backend retourne le code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Backend inaccessible: {e}")
        return False

def test_workflows_endpoint() -> bool:
    """Test 2: Vérifier l'endpoint /workflows"""
    print_info("Test 2: Vérification de l'endpoint /workflows...")
    try:
        response = requests.get(f"{BASE_URL}/workflows", timeout=5)
        if response.status_code == 200:
            workflows = response.json()
            print_success(f"Endpoint /workflows accessible ({len(workflows)} workflows trouvés)")
            return True
        else:
            print_error(f"Endpoint /workflows retourne le code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur lors de l'accès à /workflows: {e}")
        return False

def test_workflow_creation() -> Dict:
    """Test 3: Créer un workflow de test"""
    print_info("Test 3: Création d'un workflow de test...")
    
    # Simuler la création d'un workflow via l'API
    # Note: Ceci nécessite que l'agent soit exécuté manuellement
    print_warning("Ce test nécessite une exécution manuelle de l'agent")
    print_info("Veuillez exécuter un prompt simple comme 'cherche canva' dans le frontend")
    
    input("Appuyez sur Entrée une fois le workflow créé...")
    
    try:
        response = requests.get(f"{BASE_URL}/workflows", timeout=5)
        if response.status_code == 200:
            workflows = response.json()
            if len(workflows) > 0:
                latest = workflows[-1]
                print_success(f"Workflow créé: ID={latest['id']}, Prompt={latest['prompt_normalized']}")
                return latest
            else:
                print_error("Aucun workflow trouvé")
                return {}
        else:
            print_error(f"Impossible de récupérer les workflows")
            return {}
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur: {e}")
        return {}

def test_workflow_details(workflow_id: int) -> bool:
    """Test 4: Récupérer les détails d'un workflow"""
    print_info(f"Test 4: Récupération des détails du workflow {workflow_id}...")
    try:
        response = requests.get(f"{BASE_URL}/workflows/{workflow_id}", timeout=5)
        if response.status_code == 200:
            details = response.json()
            print_success(f"Détails récupérés: {len(details.get('actions', []))} actions, {len(details.get('executions', []))} exécutions")
            return True
        else:
            print_error(f"Impossible de récupérer les détails (code {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur: {e}")
        return False

def test_workflow_deactivation(workflow_id: int) -> bool:
    """Test 5: Désactiver un workflow"""
    print_info(f"Test 5: Désactivation du workflow {workflow_id}...")
    try:
        response = requests.post(f"{BASE_URL}/workflows/{workflow_id}/deactivate", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'deactivated':
                print_success("Workflow désactivé")
                return True
            else:
                print_error(f"Statut inattendu: {result}")
                return False
        else:
            print_error(f"Échec de la désactivation (code {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur: {e}")
        return False

def test_workflow_activation(workflow_id: int) -> bool:
    """Test 6: Réactiver un workflow"""
    print_info(f"Test 6: Réactivation du workflow {workflow_id}...")
    try:
        response = requests.post(f"{BASE_URL}/workflows/{workflow_id}/activate", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'activated':
                print_success("Workflow réactivé")
                return True
            else:
                print_error(f"Statut inattendu: {result}")
                return False
        else:
            print_error(f"Échec de la réactivation (code {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur: {e}")
        return False

def test_screenshot_history(session_id: str) -> bool:
    """Test 7: Vérifier l'historique des screenshots"""
    print_info(f"Test 7: Vérification de l'historique des screenshots pour la session {session_id}...")
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}/screenshots", timeout=5)
        if response.status_code == 200:
            screenshots = response.json()
            print_success(f"{len(screenshots)} screenshots trouvés")
            return True
        else:
            print_error(f"Impossible de récupérer les screenshots (code {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Erreur: {e}")
        return False

def run_all_tests():
    """Exécuter tous les tests"""
    print("\n" + "="*60)
    print("🧪 TESTS DU SYSTÈME DE WORKFLOWS")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Backend health
    results.append(("Backend Health", test_backend_health()))
    
    # Test 2: Workflows endpoint
    results.append(("Workflows Endpoint", test_workflows_endpoint()))
    
    # Test 3: Workflow creation (manuel)
    workflow = test_workflow_creation()
    if workflow:
        workflow_id = workflow.get('id')
        results.append(("Workflow Creation", True))
        
        # Test 4: Workflow details
        results.append(("Workflow Details", test_workflow_details(workflow_id)))
        
        # Test 5: Workflow deactivation
        results.append(("Workflow Deactivation", test_workflow_deactivation(workflow_id)))
        
        # Test 6: Workflow activation
        results.append(("Workflow Activation", test_workflow_activation(workflow_id)))
    else:
        results.append(("Workflow Creation", False))
        print_warning("Tests 4-6 ignorés car aucun workflow n'a été créé")
    
    # Test 7: Screenshot history (nécessite un session_id)
    print_warning("Test 7 (Screenshot History) nécessite un session_id valide")
    session_id = input("Entrez un session_id pour tester (ou appuyez sur Entrée pour ignorer): ").strip()
    if session_id:
        results.append(("Screenshot History", test_screenshot_history(session_id)))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests réussis{Colors.END}")
    
    if passed == total:
        print_success("Tous les tests sont passés ! 🎉")
    else:
        print_error(f"{total - passed} test(s) ont échoué")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()
