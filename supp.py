import random
import numpy as np
from collections import defaultdict, deque
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Literal
from multiprocessing import Pool, cpu_count

# Classe principale per l'algoritmo Tabu Search applicato al problema del flusso massimo
class MaxFlowTabuSearch:
    # Costruttore della classe
    def __init__(self, graph: Dict[Tuple[int, int], float], source: int, sink: int,
                 tabu_tenure=20, max_iterations=20000,
                 initialization: Literal['random', 'ek_partial', 'greedy'] = 'ek_partial'):
        """
        Inizializza l'istanza dell'algoritmo Tabu Search per il problema del flusso massimo.
        
        Args:
            graph: Dizionario che mappa coppie (nodo1, nodo2) alla capacità dell'arco.
            source: Nodo sorgente.
            sink: Nodo pozzo.
            tabu_tenure: Numero di iterazioni per cui una mossa rimane tabù.
            max_iterations: Numero massimo di iterazioni dell'algoritmo.
            initialization: Metodo di inizializzazione ('random', 'ek_partial', 'greedy').
        """
        # Memorizza i parametri di input
        self.source = source  # Nodo sorgente
        self.sink = sink      # Nodo pozzo
        self.graph = graph    # Grafo rappresentato come dizionario di archi e capacità
        
        # Converte gli archi in una lista indicizzata per accesso veloce
        self.edges = list(graph.keys())  # Lista di tutti gli archi (tuple)
        self.capacities = np.array([graph[e] for e in self.edges], dtype=np.float32)  # Array NumPy delle capacità
        
        # Crea una mappa inversa da arco a indice per accesso veloce
        self.edge_to_idx = {(u, v): i for i, (u, v) in enumerate(self.edges)}
        
        # Identifica gli archi che escono dalla sorgente (per calcolare il flusso totale)
        self.source_edges = [i for i, (u, v) in enumerate(self.edges) if u == self.source]
        
        # Numero totale di archi
        self.n_edges = len(self.edges)
        
        # Parametri dell'algoritmo
        self.tabu_tenure = tabu_tenure  # Durata (in iterazioni) di una mossa tabù
        self.max_iterations = max_iterations  # Numero massimo di iterazioni
        self.initialization = initialization  # Metodo di inizializzazione della soluzione
        
        # Stato dell'algoritmo (aggiornato durante la ricerca)
        self.current_flow = np.zeros(len(self.edges), dtype=np.float32)  # Flusso corrente sugli archi
        self.best_flow = np.zeros(len(self.edges), dtype=np.float32)     # Miglior flusso trovato
        self.best_value = 0.0  # Valore del miglior flusso trovato
        
        # Strutture per il monitoraggio della convergenza
        self.convergence_history = []  # Storia dei valori del miglior flusso
        self.convergence_iterations = []  # Iterazioni in cui è stato aggiornato il miglior flusso
        
        # Contatore delle valutazioni della funzione obiettivo
        self.evaluations = 0
        
        # Iterazione in cui è stato trovato il miglior flusso
        self.best_iteration = 0
        
        # Tempo di inizio dell'esecuzione
        self.start_time = time.time()
        
        # Pre-calcola la lista di adiacenza per navigare il grafo in modo efficiente
        self._precompute_adjacency()
        
        # Calcola il flusso massimo ottimale usando l'algoritmo Edmonds-Karp come riferimento
        self.optimal_max_flow_value = self._run_edmonds_karp()
        
        # Calcola le capacità totali degli archi uscenti dalla sorgente e entranti nel pozzo
        # Questi valori rappresentano un limite superiore al flusso massimo.
        self.source_out_capacity = sum(cap for (u, v), cap in self.graph.items() if u == self.source)
        self.sink_in_capacity = sum(cap for (u, v), cap in self.graph.items() if v == self.sink)
        
        # Inizializza il flusso iniziale usando il metodo specificato
        self._initialize_flow()
        
        # Dizionario che implementa la lista tabù: mappa indici di archi a iterazioni future
        self.tabu_dict = {}

    def _precompute_adjacency(self):
        """
        Pre-calcola la lista di adiacenza del grafo per un accesso più veloce.
        Crea un dizionario che mappa ogni nodo alla lista dei suoi nodi adiacenti.
        """
        self.adj = defaultdict(list)  # Dizionario con liste vuote come default
        for (u, v) in self.graph:
            # Aggiunge v alla lista dei vicini di u
            self.adj[u].append(v)

    def _run_edmonds_karp(self) -> float:
        """
        Implementa l'algoritmo di Edmonds-Karp per trovare il flusso massimo.
        Questo viene usato principalmente come riferimento per valutare la qualità della soluzione TS.
        
        Returns:
            Il valore del flusso massimo calcolato.
        """
        # Inizializza la rete residua: capacità residua per ogni arco e il suo inverso
        residual = defaultdict(float)
        for (u, v), cap in self.graph.items():
            residual[(u, v)] = cap  # Capacità residua dell'arco originale
            residual[(v, u)] = 0.0  # Capacità residua dell'arco inverso (inizialmente 0)

        flow = 0.0  # Flusso totale accumulato
        
        # Ciclo principale: continua finché trova cammini aumentanti
        while True:
            # BFS per trovare un cammino aumentante con il minor numero di archi
            parent = {}  # Mappa per ricostruire il cammino
            visited = {self.source}  # Nodi già visitati nella BFS
            queue = deque([self.source])  # Coda per la BFS
            found = False  # Flag per indicare se è stato trovato un cammino al pozzo
            
            # BFS
            while queue and not found:
                u = queue.popleft()  # Estrae il nodo corrente
                # Esplora i vicini di u
                for v in self.adj[u]:  # self.adj[u] contiene i nodi v tali che (u,v) esiste
                    # Controlla se l'arco (u,v) ha capacità residua positiva e v non è stato visitato
                    if residual[(u, v)] > 1e-6 and v not in visited:
                        visited.add(v)  # Segna v come visitato
                        parent[v] = u   # Registra u come predecessore di v
                        # Se v è il pozzo, abbiamo trovato un cammino aumentante
                        if v == self.sink:
                            found = True
                            break
                        queue.append(v)  # Aggiunge v alla coda per esplorarlo successivamente
            
            # Se non è stato trovato alcun cammino aumentante, termina
            if not found:
                break

            # Calcola il flusso massimo che si può "spingere" lungo il cammino trovato
            path_flow = float('inf')  # Inizializzato a infinito
            v = self.sink
            # Attraversa il cammino all'indietro per trovare il collo di bottiglia
            while v != self.source:
                u = parent[v]  # Predecessore di v
                # Il flusso del cammino è limitato dalla capacità residua minima
                path_flow = min(path_flow, residual[(u, v)])
                v = u  # Passa al predecessore

            # Aggiorna la rete residua lungo il cammino trovato
            v = self.sink
            while v != self.source:
                u = parent[v]
                # Sottrae il flusso dall'arco originale
                residual[(u, v)] -= path_flow
                # Aggiunge il flusso all'arco inverso (retroazione)
                residual[(v, u)] += path_flow
                v = u  # Passa al predecessore

            # Aggiunge il flusso del cammino al flusso totale
            flow += path_flow

        # Restituisce il flusso massimo totale trovato
        return flow

    def _initialize_flow(self):
        """
        Inizializza il flusso iniziale usando uno dei tre metodi specificati.
        """
        # Sceglie il metodo di inizializzazione basato sul parametro self.initialization
        if self.initialization == 'random':
            self._random_initialization()
        elif self.initialization == 'ek_partial':
            self._ek_partial_initialization()
        elif self.initialization == 'greedy':
            self._greedy_initialization()
        
        # Imposta il miglior flusso iniziale come il flusso appena generato
        self.best_value = self._calculate_flow_value(self.current_flow)
        self.best_flow = self.current_flow.copy()
        
        # Inizializza la storia della convergenza
        self.convergence_history = [self.best_value]
        self.convergence_iterations = [0]

    def _random_initialization(self):
        """
        Inizializza il flusso in modo casuale, assegnando a ogni arco un valore
        tra 0 e la sua capacità massima.
        """
        self.current_flow = np.array([random.uniform(0, cap) for cap in self.capacities], dtype=np.float32)

    def _ek_partial_initialization(self):
        """
        Inizializza il flusso usando una versione parziale dell'algoritmo di Edmonds-Karp.
        Calcola il flusso massimo ottimale, quindi lo scala di un fattore casuale (0.7-0.95).
        """
        # Inizializza un array per memorizzare il flusso ottimale trovato
        optimal_flow = np.zeros(len(self.edges), dtype=np.float32)
        
        # Crea una copia della rete residua iniziale
        residual = defaultdict(float)
        for idx, (u, v) in enumerate(self.edges):
            residual[(u, v)] = self.capacities[idx]  # Capacità iniziale
            residual[(v, u)] = 0.0                   # Capacità inversa iniziale

        # Esegue l'algoritmo di Edmonds-Karp completo sulla rete residua
        while True:
            parent = {}
            visited = {self.source}
            queue = deque([self.source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in self.adj[u]:
                    if residual[(u, v)] > 1e-6 and v not in visited:
                        visited.add(v)
                        parent[v] = u
                        if v == self.sink:
                            found = True
                            break
                        queue.append(v)
            if not found:
                break
            path_flow = float('inf')
            v = self.sink
            while v != self.source:
                u = parent[v]
                path_flow = min(path_flow, residual[(u, v)])
                v = u
            v = self.sink
            while v != self.source:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual[(v, u)] += path_flow
                # Aggiorna direttamente il flusso ottimale trovato
                optimal_flow[self.edge_to_idx[(u, v)]] += path_flow
                v = u
        
        # Scala il flusso ottimale di un fattore casuale tra 0.7 e 0.95
        # Questo fornisce un buon punto di partenza che è una soluzione valida ma non ottimale
        factor = random.uniform(0.7, 0.95)
        self.current_flow = optimal_flow * factor

    def _greedy_initialization(self):
        """
        Inizializza il flusso usando un approccio greedy.
        Trova iterativamente cammini aumentanti "migliori" e invia flusso lungo di essi.
        """
        # Inizializza il flusso corrente a zero su tutti gli archi
        self.current_flow = np.zeros(len(self.edges), dtype=np.float32)
        
        # Crea una copia delle capacità residue iniziali
        residual = self.capacities.copy()
        
        # Cerca fino a 10 cammini aumentanti greedy
        for _ in range(10): 
            # Trova un cammino aumentante usando una strategia greedy
            path = self._find_augmenting_path_greedy(residual)
            
            # Se non trova un cammino, termina
            if not path:
                break
            
            # Calcola il flusso massimo che si può inviare lungo il cammino trovato
            # È il minimo delle capacità residue degli archi nel cammino
            min_residual = min(residual[edge_idx] for edge_idx in path)
            
            # Aggiorna il flusso corrente e le capacità residue
            for edge_idx in path:
                self.current_flow[edge_idx] += min_residual  # Aumenta il flusso sull'arco
                residual[edge_idx] -= min_residual         # Riduce la capacità residua

    def _find_augmenting_path_greedy(self, residual):
        """
        Trova un cammino aumentante usando una strategia greedy.
        A differenza della BFS (che trova il cammino con meno archi), questa
        cerca di trovare un cammino che permetta di inviare il massimo flusso possibile.
        
        Args:
            residual: Array delle capacità residue sugli archi.
            
        Returns:
            Una lista di indici di archi che formano il cammino aumentante, o None se non trovato.
        """
        # Strutture per la ricerca
        parent = {}          # Mappa per ricostruire il cammino
        capacity = {}        # Capacità massima che si può portare a ciascun nodo
        visited = set()      # Nodi già visitati
        queue = [self.source]  # Lista di nodi da espandere (non una coda FIFO!)
        
        # Inizializzazione
        visited.add(self.source)
        capacity[self.source] = float('inf')  # La sorgente può "portare" flusso infinito
        
        # Ciclo principale della ricerca greedy
        while queue:
            # Sceglie il nodo nella coda con la capacità massima disponibile
            # Questo è il punto chiave della strategia greedy
            u = max(queue, key=lambda x: capacity[x])
            queue.remove(u)  # Rimuove il nodo selezionato dalla lista
            
            # Esplora i vicini di u
            for v in self.adj[u]:
                edge_idx = self.edge_to_idx[(u, v)]  # Indice dell'arco (u,v)
                
                # Controlla se l'arco è ammissibile (capacità residua > 0) e v non è stato visitato
                if residual[edge_idx] > 1e-6 and v not in visited:
                    visited.add(v)  # Segna v come visitato
                    parent[v] = u   # Registra u come predecessore di v
                    
                    # Calcola la capacità massima che si può portare fino a v
                    # È il minimo tra la capacità che arrivava a u e la capacità dell'arco (u,v)
                    capacity[v] = min(capacity[u], residual[edge_idx])
                    
                    # Se v è il pozzo, abbiamo trovato un cammino aumentante
                    if v == self.sink:
                        # Ricostruisce il cammino dal pozzo alla sorgente seguendo i predecessori
                        path = []
                        node = self.sink
                        while node != self.source:
                            # Trova l'indice dell'arco dal predecessore al nodo corrente
                            edge_idx = self.edge_to_idx[(parent[node], node)]
                            path.append(edge_idx)  # Aggiunge l'indice dell'arco al cammino
                            node = parent[node]    # Passa al predecessore
                        # Inverte il cammino per andare dalla sorgente al pozzo e lo restituisce
                        return path[::-1]
                    
                    # Se v non è il pozzo, lo aggiunge alla lista per essere esplorato
                    queue.append(v)
        
        # Se la coda si svuota senza trovare il pozzo, non esiste un cammino aumentante
        return None

    def _calculate_flow_value(self, flow):
        """
        Calcola il valore totale del flusso, sommando i flussi sugli archi uscenti dalla sorgente.
        
        Args:
            flow: Array NumPy contenente il flusso su ciascun arco.
            
        Returns:
            Il valore totale del flusso dalla sorgente al pozzo.
        """
        self.evaluations += 1  # Incrementa il contatore delle valutazioni
        # Somma i flussi sugli archi che escono dalla sorgente
        return np.sum(flow[self.source_edges])

    def _find_augmenting_path(self, flow):
        """
        Trova un cammino aumentante nella rete residua usando una BFS.
        Questa versione lavora con un flusso corrente specifico, non con capacità residue iniziali.
        
        Args:
            flow: Il flusso corrente sugli archi.
            
        Returns:
            Una lista di tuple (indice_arco, direzione) che rappresenta il cammino aumentante,
            o None se non trovato.
        """
        # Strutture per la BFS
        parent = {}       # Mappa per ricostruire il cammino
        visited = set()   # Nodi già visitati
        queue = deque([self.source])  # Coda per la BFS
        visited.add(self.source)
        found = False     # Flag per indicare se è stato trovato il pozzo
        
        # BFS per trovare un cammino aumentante
        while queue and not found:
            u = queue.popleft()  # Estrae il nodo corrente
            
            # Esplora i vicini di u
            for v in self.adj[u]:
                # Calcola la capacità residua dell'arco (u,v)
                # Capacità residua = capacità totale - flusso attuale
                residual = self.graph[(u, v)] - flow[self.edge_to_idx[(u, v)]]
                
                # Controlla se l'arco ha capacità residua positiva e v non è stato visitato
                if residual > 1e-6 and v not in visited:
                    visited.add(v)
                    parent[v] = u
                    
                    # Se v è il pozzo, abbiamo trovato un cammino aumentante
                    if v == self.sink:
                        found = True
                        break
                    
                    queue.append(v)  # Aggiunge v alla coda per esplorarlo
        
        # Se non è stato trovato un cammino aumentante, restituisce None
        if not found:
            return None

        # Ricostruisce il cammino aumentante trovato
        path = []
        v = self.sink
        # Attraversa il cammino all'indietro dalla sorgente al pozzo
        while v != self.source:
            u = parent[v]
            edge_idx = self.edge_to_idx[(u, v)]  # Indice dell'arco
            
            # Aggiunge una tupla (indice_arco, direzione) al cammino
            # La direzione è True se l'arco è percorso nel verso originale (da u a v)
            path.append((edge_idx, True))
            v = u  # Passa al predecessore
        
        # Inverte il cammino per andare dalla sorgente al pozzo e lo restituisce
        return path[::-1]

    def _generate_edge_exchange_moves(self, current_flow, n_exchanges=5):
        """
        Genera mosse di scambio di flusso tra archi saturi e non saturi.
        Questo tipo di mossa cerca di ridistribuire il flusso per migliorare la soluzione.
        
        Args:
            current_flow: Il flusso corrente sugli archi.
            n_exchanges: Numero massimo di mosse di scambio da generare.
            
        Returns:
            Una lista di mosse di scambio.
        """
        moves = []  # Lista per memorizzare le mosse generate
        
        # Identifica gli archi saturi (flusso >= capacità) e non saturi
        saturated = [i for i in range(self.n_edges) 
                    if current_flow[i] >= self.capacities[i] - 1e-6]
        unsaturated = [i for i in range(self.n_edges) 
                      if current_flow[i] < self.capacities[i] - 1e-6]
        
        # Genera fino a n_exchanges mosse di scambio
        for _ in range(min(n_exchanges, len(saturated), len(unsaturated))):
            # Sceglie casualmente un arco saturo (sorgente del flusso) e uno non saturo (destinazione)
            src, dest = random.choice(saturated), random.choice(unsaturated)
            
            # Calcola la quantità di flusso massima che si può scambiare
            # È il minimo tra il flusso sull'arco sorgente e lo spazio disponibile sull'arco destinazione
            delta = min(current_flow[src], self.capacities[dest] - current_flow[dest])
            
            # Se c'è flusso da scambiare, crea la mossa
            if delta > 1e-6:
                # Crea una nuova configurazione del flusso
                new_flow = current_flow.copy()
                new_flow[src] -= delta  # Riduce il flusso sull'arco sorgente
                new_flow[dest] += delta # Aumenta il flusso sull'arco destinazione
                
                # Aggiunge la mossa alla lista: (nuovo_flusso, (indice_sorgente, indice_destinazione), quantità, è_tabù)
                moves.append((new_flow, (src, dest), delta, False))
        
        return moves

    def _get_neighborhood_moves(self, current_flow, iteration):
        """
        Genera un insieme di mosse candidate (vicini) della soluzione corrente.
        Combina diverse strategie per esplorare il vicinato.
        
        Args:
            current_flow: Il flusso corrente sugli archi.
            iteration: L'iterazione corrente dell'algoritmo.
            
        Returns:
            Una lista di mosse candidate.
        """
        moves = []  # Lista per memorizzare le mosse generate
        
        # 1. Mosse di incremento/decremento su un sottoinsieme casuale di archi
        # Sceglie un numero limitato di archi casuali da modificare (per efficienza)
        sample_size = min(30, self.n_edges // 80)
        for edge_idx in random.sample(range(self.n_edges), sample_size):
            curr = current_flow[edge_idx]  # Flusso corrente sull'arco
            cap = self.capacities[edge_idx]  # Capacità massima dell'arco
            
            # Se l'arco non è saturo, genera una mossa per incrementare il flusso
            if curr < cap - 1e-6:
                # Calcola l'incremento (max 20% della capacità)
                step = min(cap - curr, cap * 0.2)
                new_flow = current_flow.copy()
                new_flow[edge_idx] += step  # Incrementa il flusso
                
                # Controlla se questa mossa è tabù (basandosi solo sull'arco modificato)
                is_tabu = edge_idx in self.tabu_dict and self.tabu_dict[edge_idx] > iteration
                
                # Aggiunge la mossa alla lista: (nuovo_flusso, indice_arco, incremento, è_tabù)
                moves.append((new_flow, edge_idx, step, is_tabu))
            
            # Se l'arco ha flusso > 0, genera una mossa per decrementare il flusso
            if curr > 1e-6:
                # Calcola il decremento (max 20% della capacità)
                step = min(curr, cap * 0.2)
                new_flow = current_flow.copy()
                new_flow[edge_idx] -= step  # Decrementa il flusso
                
                # Controlla se questa mossa è tabù
                is_tabu = edge_idx in self.tabu_dict and self.tabu_dict[edge_idx] > iteration
                
                # Aggiunge la mossa alla lista: (nuovo_flusso, indice_arco, -decremento, è_tabù)
                moves.append((new_flow, edge_idx, -step, is_tabu))

        # 2. Ogni 20 iterazioni, aggiunge mosse di scambio di flusso
        if iteration % 20 == 0:
            moves.extend(self._generate_edge_exchange_moves(current_flow))

        # 3. Ogni 100 iterazioni, cerca un cammino aumentante e genera una mossa lungo di esso
        if iteration % 100 == 0:
            # Trova un cammino aumentante nella rete residua corrente
            path = self._find_augmenting_path(current_flow)
            
            # Se trova un cammino, genera una mossa che invia flusso lungo quel cammino
            if path:
                # Calcola il flusso massimo che si può inviare lungo il cammino
                # È il minimo delle capacità residue degli archi nel cammino
                min_residual = min(
                    self.capacities[edge_idx] - current_flow[edge_idx] if is_forward 
                    else current_flow[edge_idx]
                    for edge_idx, is_forward in path
                )
                
                # Se c'è spazio per inviare flusso, crea la mossa
                if min_residual > 1e-6:
                    new_flow = current_flow.copy()
                    # Aggiorna il flusso su ciascun arco del cammino
                    for edge_idx, is_forward in path:
                        if is_forward:
                            new_flow[edge_idx] += min_residual  # Incrementa se percorso in avanti
                        else:
                            new_flow[edge_idx] -= min_residual  # Decrementa se percorso all'indietro
                    
                    # Aggiunge la mossa alla lista: (nuovo_flusso, tuple_di_indici_archi, flusso_inviato, è_tabù)
                    moves.append((new_flow, tuple(edge_idx for edge_idx, _ in path), min_residual, False))
        
        return moves

    def _diversify_solution(self, current_flow):
        """
        Applica una diversificazione alla soluzione corrente.
        Riduce parzialmente il flusso e aggiunge rumore casuale per spostare la ricerca.
        
        Args:
            current_flow: Il flusso corrente da diversificare.
            
        Returns:
            Una nuova configurazione del flusso diversificata.
        """
        # Sceglie un fattore casuale per ridurre il flusso (tra 0.6 e 0.8)
        reset_factor = random.uniform(0.6, 0.8)
        
        # Sceglie un fattore casuale per il rumore (tra 0.1 e 0.2)
        noise_factor = random.uniform(0.1, 0.2)
        
        # Riduce tutti i flussi di un fattore (1 - reset_factor)
        # Esempio: se reset_factor = 0.7, new_flow diventa circa il 30% di current_flow
        new_flow = current_flow * (1 - reset_factor)
        
        # Sceglie un numero casuale di archi su cui applicare rumore
        num_perturb = min(len(self.edges) // 20, 5)
        indices = random.sample(range(len(self.edges)), num_perturb)
        
        # Aggiunge rumore casuale al flusso su alcuni archi selezionati
        for idx in indices:
            cap = self.capacities[idx]  # Capacità dell'arco
            # Genera rumore casuale (tra -noise_factor*cap e +noise_factor*cap)
            noise = random.uniform(-noise_factor, noise_factor) * cap
            # Aggiorna il flusso sull'arco, assicurandosi che rimanga nei limiti [0, capacità]
            new_flow[idx] = max(0, min(cap, new_flow[idx] + noise))
        
        return new_flow

    def _adaptive_parameters(self, iteration, stagnation_counter):
        """
        Adatta i parametri dell'algoritmo in base al comportamento della ricerca.
        
        Args:
            iteration: L'iterazione corrente.
            stagnation_counter: Il numero di iterazioni consecutive senza miglioramenti.
            
        Returns:
            True se è stata applicata una diversificazione, False altrimenti.
        """
        # Se la ricerca è stagnante da molto tempo, aumenta la tenure tabù
        # per favorire l'intensificazione o aiutare a uscire da minimi locali
        if stagnation_counter > 100:
            self.tabu_tenure = min(int(self.tabu_tenure * 1.2), 150)
        
        # Se la ricerca non è stagnante, diminuisci la tenure tabù
        # per permettere una maggiore esplorazione
        elif stagnation_counter < 50:
            self.tabu_tenure = max(int(self.tabu_tenure * 0.9), 10)
        
        # Ogni 2000 iterazioni, se la stagnazione è molto lunga, applica diversificazione
        if iteration % 2000 == 0 and stagnation_counter > 300:
            self.current_flow = self._diversify_solution(self.current_flow)
            return True  # Indica che è stata applicata una diversificazione
        
        return False  # Nessuna diversificazione applicata

    def plot_convergence(self, filename=None, title_suffix=""):
        """
        Crea un grafico della convergenza dell'algoritmo.
        
        Args:
            filename: Nome del file in cui salvare il grafico (se None, mostra il grafico).
            title_suffix: Suffixo da aggiungere al titolo del grafico.
        """
        plt.figure(figsize=(12, 7))
        
        # Disegna la storia della convergenza del miglior flusso trovato
        plt.plot(self.convergence_iterations, self.convergence_history, 'b-', label='Best Flow Value', linewidth=2)
        
        # Disegna una linea orizzontale per il flusso massimo ottimale (riferimento)
        plt.axhline(y=self.optimal_max_flow_value, color='r', linestyle='--',
                   linewidth=1.5, label=f'Optimal (EK): {self.optimal_max_flow_value:.2f}')
        
        # Evidenzia il punto del miglior flusso trovato
        plt.scatter([self.best_iteration], [self.best_value], color='g',
                   s=100, zorder=5, label=f'Best: {self.best_value:.2f} at {self.best_iteration}')
        
        # Imposta il titolo e le etichette
        plt.title(f'Tabu Search Convergence {title_suffix}\n'
                 f'Nodes: {len(set(n for n, _ in self.graph))}, Edges: {len(self.graph)}', pad=20)
        plt.xlabel('Iterations', labelpad=10)
        plt.ylabel('Flow Value', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Aggiunge informazioni sulle performance
        info_text = (f"Optimal: {self.optimal_max_flow_value:.2f}\n"
                     f"Evaluations: {self.evaluations}\n"
                     f"Time: {time.time() - self.start_time:.2f}s")
        plt.gcf().text(0.15, 0.7, info_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Salva o mostra il grafico
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def search(self, seed=None):
        """
        Esegue l'algoritmo principale di Tabu Search.
        
        Args:
            seed: Seme per il generatore di numeri casuali (per riproducibilità).
            
        Returns:
            Un dizionario contenente i risultati della ricerca.
        """
        # Imposta il seme casuale se fornito
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Re-inizializza la lista tabù
        self.tabu_dict = {}
        
        # Contatore per il numero di iterazioni consecutive senza miglioramenti
        stagnation_counter = 0
        
        # Calcola un limite superiore teorico al flusso massimo
        upper_bound = min(self.source_out_capacity, self.sink_in_capacity)
        
        # Ciclo principale dell'algoritmo
        for iteration in range(self.max_iterations):
            # Adatta i parametri e applica diversificazione se necessario
            diversified = self._adaptive_parameters(iteration, stagnation_counter)
            if diversified:
                stagnation_counter = 0  # Resetta il contatore di stagnazione
            
            # Genera l'insieme delle mosse candidate (vicinato)
            moves = self._get_neighborhood_moves(self.current_flow, iteration)
            
            # Se non ci sono mosse valide, incrementa il contatore di stagnazione e continua
            if not moves:
                stagnation_counter += 1
                continue
            
            # Seleziona la migliore mossa ammessa
            best_move = None
            best_val = -np.inf
            
            # Esamina tutte le mosse candidate
            for move_flow, idx, delta, is_tabu in moves:
                # Una mossa è ammessa se:
                # 1. Non è tabù, OPPURE
                # 2. È tabù ma porta a una soluzione migliore del miglior valore trovato finora
                # (Questo è il criterio di aspirazione)
                if not is_tabu or self._calculate_flow_value(move_flow) > self.best_value:
                    val = self._calculate_flow_value(move_flow)  # Valore della soluzione della mossa
                    # Se questa mossa porta a un valore migliore delle altre mosse ammesse, selezionala
                    if val > best_val:
                        best_val = val
                        best_move = (move_flow, idx)  # La mossa è una tupla (nuovo_flusso, indice_arco/i)
            
            # Se nessuna mossa ammessa è stata trovata, incrementa il contatore di stagnazione e continua
            if best_move is None:
                stagnation_counter += 1
                continue
            
            # Applica la migliore mossa selezionata
            new_flow, edge_idx = best_move
            self.current_flow = new_flow
            
            # Aggiorna la lista tabù: rende tabù l'arco (o gli archi) modificato/i dalla mossa
            # L'arco rimane tabù per self.tabu_tenure iterazioni
            if isinstance(edge_idx, tuple):  # Mossa su multipli archi (es: path o scambio)
                for e in edge_idx:
                    self.tabu_dict[e] = iteration + self.tabu_tenure
            else:  # Mossa su singolo arco
                self.tabu_dict[edge_idx] = iteration + self.tabu_tenure
            
            # Periodicamente (ogni 500 iterazioni), pulisce la lista tabù
            # Rimuove le voci relative ad archi che non sono più tabù
            if iteration % 500 == 0:
                self.tabu_dict = {k: v for k, v in self.tabu_dict.items() if v > iteration}
            
            # Calcola il valore del flusso corrente
            current_val = self._calculate_flow_value(self.current_flow)
            
            # Se il flusso corrente è migliore del miglior flusso trovato finora, aggiorna il record
            if current_val > self.best_value:
                self.best_value = current_val
                self.best_flow = self.current_flow.copy()
                self.best_iteration = iteration
                stagnation_counter = 0  # Resetta il contatore di stagnazione
            else:
                # Altrimenti, incrementa il contatore di stagnazione
                stagnation_counter += 1
            
            # Aggiorna la storia della convergenza
            if iteration % 100 == 0 or current_val > self.convergence_history[-1]:
                self.convergence_history.append(self.best_value)
                self.convergence_iterations.append(iteration)
            
            # Criterio di arresto: se il flusso corrente è molto vicino al limite superiore teorico
            if abs(current_val - upper_bound) < 1e-6:
                break
        
        # Calcola il tempo totale di esecuzione
        elapsed_time = time.time() - self.start_time
        
        # Restituisce un dizionario con i risultati finali
        return {
            'best_value': self.best_value,      # Il miglior valore di flusso trovato
            'best_flow': self.best_flow.tolist(), # La configurazione del flusso corrispondente
            'best_iteration': self.best_iteration, # L'iterazione in cui è stato trovato il miglior flusso
            'evaluations': self.evaluations,    # Numero totale di valutazioni della funzione obiettivo
            'elapsed_time': elapsed_time,       # Tempo totale di esecuzione
            'convergence_history': self.convergence_history, # Storia dei valori del miglior flusso
            'convergence_iterations': self.convergence_iterations, # Iterazioni corrispondenti alla storia
            'optimal_flow': self.optimal_max_flow_value # Flusso massimo ottimale di riferimento (Edmonds-Karp)
        }

# Funzioni di supporto per la gestione dei file e degli esperimenti

def read_instance(filename: str) -> Tuple[Dict[Tuple[int, int], float], int, int, int, int]:
    """
    Legge un'istanza del problema da un file di testo.
    
    Args:
        filename: Nome del file contenente l'istanza.
        
    Returns:
        Una tupla contenente (grafo, sorgente, pozzo, numero_nodi, numero_archi).
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Le prime 4 righe contengono metadati
    n_nodes = int(lines[0])  # Numero di nodi
    n_edges = int(lines[1])  # Numero di archi
    source = int(lines[2])   # Nodo sorgente
    sink = int(lines[3])     # Nodo pozzo
    
    # Le righe successive contengono gli archi
    graph = {}
    for i in range(4, min(4 + n_edges, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 3:
            u, v, capacity = int(parts[0]), int(parts[1]), float(parts[2])
            graph[(u, v)] = capacity  # Aggiunge l'arco al grafo
    
    return graph, source, sink, n_nodes, n_edges

def run_single_experiment(args):
    """
    Esegue un singolo esperimento (una esecuzione dell'algoritmo).
    
    Args:
        args: Tupla contenente i parametri dell'esperimento.
        
    Returns:
        I risultati dell'esperimento.
    """
    # Estrae i parametri dagli argomenti
    graph, source, sink, seed, output_dir, filename_base, run = args
    
    # Crea un'istanza dell'algoritmo e lo esegue
    solver = MaxFlowTabuSearch(graph, source, sink)
    result = solver.search(seed=seed)
    
    # Crea e salva il grafico di convergenza
    plot_path = os.path.join(output_dir, f"{filename_base}_run{run+1}_convergence.png")
    solver.plot_convergence(plot_path, title_suffix=f"(Run {run+1})")
    
    # Stampa i risultati dell'esecuzione
    print(f"Run {run + 1}: Best={result['best_value']:.2f}, Iter={result['best_iteration']}, "
          f"Eval={result['evaluations']}, Time={result['elapsed_time']:.4f}s")
    
    return result

def compute_mean_convergence(runs_results, max_iterations):
    """
    Calcola la convergenza media su più esecuzioni.
    
    Args:
        runs_results: Lista dei risultati delle esecuzioni.
        max_iterations: Numero massimo di iterazioni.
        
    Returns:
        Tuple di array (iterazioni_comuni, flusso_medio, deviazione_standard).
    """
    # Crea un insieme comune di iterazioni per interpolare i dati
    common_iterations = np.linspace(0, max_iterations, 1000)
    all_flows = []
    
    # Interpola i dati di ciascuna esecuzione sulle iterazioni comuni
    for run in runs_results:
        interp_flow = np.interp(
            common_iterations,
            run['convergence_iterations'],
            run['convergence_history'],
            right=run['convergence_history'][-1]  # Usa l'ultimo valore per extrapolazione
        )
        all_flows.append(interp_flow)
    
    # Calcola la media e la deviazione standard dei flussi
    mean_flow = np.mean(all_flows, axis=0)
    std_flow = np.std(all_flows, axis=0)
    
    return common_iterations, mean_flow, std_flow

def get_best_run(runs_results):
    """
    Trova l'esecuzione migliore tra quelle eseguite.
    La "migliore" è quella con il miglior valore di flusso, con una piccola penalità per il tempo.
    
    Args:
        runs_results: Lista dei risultati delle esecuzioni.
        
    Returns:
        I risultati dell'esecuzione migliore.
    """
    # Calcola uno score per ciascuna esecuzione
    scores = [r['best_value'] - (0.0001 * r['best_iteration']) for r in runs_results]
    best_idx = np.argmax(scores)  # Trova l'indice dell'esecuzione con score massimo
    return runs_results[best_idx]  # Restituisce i risultati dell'esecuzione migliore

def plot_average_convergence(results, filename, optimal_value):
    """
    Crea un grafico della convergenza media su più esecuzioni.
    
    Args:
        results: Lista dei risultati delle esecuzioni.
        filename: Nome del file in cui salvare il grafico.
        optimal_value: Valore del flusso massimo ottimale.
    """
    plt.figure(figsize=(12, 7))
    
    # Calcola la convergenza media
    max_iter = max([r['convergence_iterations'][-1] for r in results])
    x_vals, mean_conv, std_conv = compute_mean_convergence(results, max_iter)
    
    # Disegna la convergenza media e la banda di deviazione standard
    plt.plot(x_vals, mean_conv, 'b-', linewidth=2, label='Average Flow')
    plt.fill_between(x_vals, mean_conv - std_conv, mean_conv + std_conv, alpha=0.2)
    
    # Disegna una linea per il valore ottimale
    plt.axhline(y=optimal_value, color='r', linestyle='--', linewidth=1.5, label=f'Optimal (EK): {optimal_value:.2f}')
    
    # Imposta titolo ed etichette
    plt.title(f'Average Convergence for {os.path.basename(filename)}\n{len(results)} runs', pad=20)
    plt.xlabel('Iterations', labelpad=10)
    plt.ylabel('Flow Value', labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Salva il grafico
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_experiments(output_dir="results"):
    """
    Esegue una serie di esperimenti su più istanze del problema.
    
    Args:
        output_dir: Directory in cui salvare i risultati.
        
    Returns:
        Un dizionario contenente i risultati aggregati.
    """
    # Crea la directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Trova tutti i file di istanza nella directory corrente
    instance_files = [f for f in os.listdir() if f.startswith("network_") and f.endswith(".txt")]
    
    results = {}  # Dizionario per memorizzare i risultati
    
    # Processa ciascun file di istanza
    for filename in sorted(instance_files):
        print(f"\n--- Processing instance: {filename} ---")
        try:
            # Legge l'istanza dal file
            graph, source, sink, n_nodes, n_edges = read_instance(filename)
            
            # Prepara gli argomenti per 10 esecuzioni indipendenti
            args_list = [(graph, source, sink, run + 42, output_dir,
                         os.path.splitext(os.path.basename(filename))[0], run) for run in range(10)]
            
            # Esegue le esecuzioni in parallelo (usando metà dei core CPU disponibili)
            with Pool(max(1, cpu_count() // 2)) as p:
                runs_results = p.map(run_single_experiment, args_list)
            
            # Calcola e salva il grafico di convergenza media
            max_iter = max([r['convergence_iterations'][-1] for r in runs_results])
            x_vals, mean_conv, std_conv = compute_mean_convergence(runs_results, max_iter)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, mean_conv, label='Mean Flow', color='blue')
            plt.axhline(y=runs_results[0]['optimal_flow'], color='red',
                       linestyle='--', label='Optimal (Edmonds-Karp)')
            plt.title(f'Mean Convergence for {filename}')
            plt.xlabel('Iterations')
            plt.ylabel('Flow Value')
            plt.grid(True)
            plt.legend()
            
            mean_plot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mean_convergence.png")
            plt.savefig(mean_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Trova e salva il grafico dell'esecuzione migliore
            best_run = get_best_run(runs_results)
            
            plt.figure(figsize=(10, 6))
            plt.plot(best_run['convergence_iterations'], best_run['convergence_history'],
                    label=f'Best Run (Flow={best_run["best_value"]:.2f})', color='blue')
            plt.axhline(y=best_run['optimal_flow'], color='red',
                       linestyle='--', label='Optimal')
            plt.title(f'Best Run Convergence for {filename}\n'
                     f'Final Flow: {best_run["best_value"]:.2f} | '
                     f'Optimal: {best_run["optimal_flow"]:.2f} | '
                     f'Iter: {best_run["best_iteration"]}')
            plt.xlabel('Iterations')
            plt.ylabel('Flow Value')
            plt.grid(True)
            plt.legend()
            
            best_plot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_best_convergence.png")
            plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calcola le statistiche aggregate
            all_flows = [r['best_value'] for r in runs_results]
            result = {
                'best': max(all_flows),  # Miglior flusso tra tutte le esecuzioni
                'mean': np.mean(all_flows),  # Flusso medio
                'std': np.std(all_flows),    # Deviazione standard
                'avg_iterations': np.mean([r['best_iteration'] for r in runs_results]),  # Iterazioni medie
                'avg_evaluations': np.mean([r['evaluations'] for r in runs_results]),    # Valutazioni medie
                'avg_time': np.mean([r['elapsed_time'] for r in runs_results]),          # Tempo medio
                'optimal_flow': runs_results[0]['optimal_flow']  # Flusso ottimale (uguale per tutte)
            }
            
            results[filename] = result  # Aggiunge i risultati al dizionario
            
            # Stampa le statistiche finali
            print(f"\nFinal statistics for {filename}:")
            print(f"Best Flow (TS): {result['best']:.2f} (Optimal: {result['optimal_flow']:.2f})")
            print(f"Mean Flow (TS): {result['mean']:.2f} ± {result['std']:.2f}")
            print(f"Avg Iterations: {result['avg_iterations']:.0f}")
            print(f"Avg Evaluations: {result['avg_evaluations']:.0f}")
            print(f"Avg Time: {result['avg_time']:.4f}s")
            
        except Exception as e:
            # Gestisce eventuali errori durante l'elaborazione di un'istanza
            print(f"Error processing {filename}: {str(e)}")
    
    return results  # Restituisce tutti i risultati

# Punto di ingresso dello script
if __name__ == "__main__":
    # Esegue gli esperimenti e memorizza i risultati
    experiment_results = run_experiments()
