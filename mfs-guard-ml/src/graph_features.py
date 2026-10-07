"""Incremental directed graph and union-find, observed before inserting edges."""
from collections import defaultdict


class HistoricalGraph:
    def __init__(self):
        self.incoming = defaultdict(set)
        self.outgoing = defaultdict(set)
        self.parent = {}
        self.size = {}

    def root(self, node):
        if node not in self.parent:
            self.parent[node], self.size[node] = node, 1
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def observe(self, u, r):
        inc, out = self.incoming, self.outgoing
        return {'sender_in_degree': len(inc[u]), 'sender_out_degree': len(out[u]),
            'receiver_in_degree': len(inc[r]), 'receiver_out_degree': len(out[r]),
            'unique_sender_count': len(inc[r]), 'unique_receiver_count': len(out[u]),
            'connected_component_size': self.size[self.root(r)],
            'reciprocal_transfer_ratio': len(inc[r] & out[r]) / max(1, len(out[r])),
            'shared_recipient_count': len(out[u] & out[r])}

    def update(self, u, r):
        self.incoming[r].add(u)
        self.outgoing[u].add(r)
        a, b = self.root(u), self.root(r)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
