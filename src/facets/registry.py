import json
from typing import List, Dict, Any, Optional

class FacetRegistry:
    def __init__(self, registry_file: str):
        self.registry_file = registry_file
        self.facets = []
        self.load()
        
    def load(self):
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.facets = data.get("facets", [])
            
    def get_all_facets(self) -> List[Dict[str, Any]]:
        return self.facets
        
    def get_facets_by_group(self, group: str) -> List[Dict[str, Any]]:
        return [f for f in self.facets if f.get("group") == group]
        
    def get_facet(self, facet_id: str) -> Optional[Dict[str, Any]]:
        for f in self.facets:
            if f.get("facet_id") == facet_id:
                return f
        return None
