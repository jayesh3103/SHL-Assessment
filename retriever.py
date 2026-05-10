import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CatalogRetriever:
    def __init__(self, catalog_path: str = "shl_product_catalog.json"):
        self.catalog = []
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        self._load_catalog(catalog_path)
    
    def _load_catalog(self, catalog_path: str):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            raw_data = json.loads(f.read(), strict=False)
            
        for item in raw_data:
            keys = item.get("keys", [])
            test_type = keys[0][0] if keys else "U"
            
            cleaned_item = {
                "name": item.get("name", ""),
                "url": item.get("link", ""),
                "test_type": test_type,
                "job_levels": item.get("job_levels", []),
                "description": item.get("description", ""),
                "keys": keys
            }
            self.catalog.append(cleaned_item)
            
            # Create a searchable document string
            doc_parts = [
                cleaned_item["name"],
                cleaned_item["description"],
                " ".join(cleaned_item["job_levels"]),
                " ".join(cleaned_item["keys"])
            ]
            self.documents.append(" ".join(doc_parts))
            
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            
    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Search the catalog using TF-IDF cosine similarity.
        """
        if not query.strip() or self.tfidf_matrix is None:
            return self.catalog[:top_k]
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # If similarity is 0, we can still return it, but maybe better to filter out entirely irrelevant?
            # We'll just return top k for now.
            results.append(self.catalog[idx])
            
        return results

# Singleton instance to be used across the app
retriever = CatalogRetriever()
