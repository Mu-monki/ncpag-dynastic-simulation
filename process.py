import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

class DynastyDilutionSimulator:
    """
    Simulate and compare the impact of anti-dynasty bills with varying
    consanguinity limits on political power concentration.
    """

    def __init__(self, candidates_file, relationships_file):
        """
        Load candidates and family relationships.

        Parameters
        ----------
        candidates_file : str
            CSV with columns: id, name, province, position, elected (0/1), year
        relationships_file : str
            CSV with columns: person1_id, person2_id, degree
            degree is an integer representing the consanguinity level.
        """
        self.candidates = pd.read_csv(candidates_file)
        self.relationships = pd.read_csv(relationships_file)

        # Build a graph of family ties
        self.G = nx.Graph()
        for _, row in self.relationships.iterrows():
            self.G.add_edge(row['person1_id'], row['person2_id'], degree=row['degree'])

    def get_relatives_bfs(self, person_id, max_degree):
        """
        Find all relatives within a given degree limit using BFS.
        NOTE: This is a simplified approach. A robust solution would require a
        proper genealogical tree to accurately calculate cumulative degree.
        """
        if person_id not in self.G:
            return set()

        visited = {person_id: 0}
        queue = [person_id]
        relatives = set()

        while queue:
            current = queue.pop(0)
            for neighbor in self.G.neighbors(current):
                # Simplified degree accumulation (assumes additive path)
                edge_degree = self.G[current][neighbor]['degree']
                new_degree = visited[current] + edge_degree
                if neighbor not in visited and new_degree <= max_degree:
                    visited[neighbor] = new_degree
                    relatives.add(neighbor)
                    queue.append(neighbor)
        return relatives

    def analyze_dynastic_power(self, year, province, degree_limit):
        """
        Analyze the political landscape under a specific dynasty ban.

        Returns:
        - dynastic_candidates_df: DataFrame of candidates who would be disqualified.
        - stats: Dictionary of key metrics (seats affected, HHI, etc.).
        - family_clusters: List of families (connected components) under this limit.
        """
        # Filter candidates for the specified election and province
        mask = (self.candidates['year'] == year) & (self.candidates['province'] == province)
        local_candidates = self.candidates[mask].copy()
        if local_candidates.empty:
            return None, None, None

        candidate_ids = set(local_candidates['id'])
        elected_ids = set(local_candidates[local_candidates['elected'] == 1]['id'])

        # Build a subgraph of only these candidates
        subG = self.G.subgraph(candidate_ids).copy()
        for cid in candidate_ids:
            if cid not in subG:
                subG.add_node(cid)

        # Find dynastic candidates (those with a relative also running)
        dynastic_ids = set()
        for cid in candidate_ids:
            relatives = self.get_relatives_bfs(cid, degree_limit)
            running_relatives = relatives & candidate_ids
            if running_relatives:
                dynastic_ids.add(cid)

        dynastic_df = local_candidates[local_candidates['id'].isin(dynastic_ids)]

        # Find connected components (potential families) within this limit
        # This is a simplification; in reality, families are defined by the degree limit.
        family_clusters = list(nx.connected_components(subG))

        # Calculate Political HHI based on these families
        seats_per_family = []
        for fam in family_clusters:
            fam_elected = len([cid for cid in fam if cid in elected_ids])
            seats_per_family.append(fam_elected)

        total_seats = sum(seats_per_family)
        if total_seats > 0:
            hhi = sum((s/total_seats)**2 for s in seats_per_family)
        else:
            hhi = 0

        # Compile statistics
        stats = {
            'year': year,
            'province': province,
            'degree_limit': degree_limit,
            'total_candidates': len(local_candidates),
            'total_seats': total_seats,
            'dynastic_candidates': len(dynastic_df),
            'pct_candidates_affected': round(len(dynastic_df)/len(local_candidates)*100, 2) if len(local_candidates) > 0 else 0,
            'dynastic_elected': dynastic_df['elected'].sum() if not dynastic_df.empty else 0,
            'pct_seats_affected': round(dynastic_df['elected'].sum()/total_seats*100, 2) if total_seats > 0 else 0,
            'political_hhi': round(hhi, 4),
            'num_families': len(family_clusters)
        }

        return dynastic_df, stats, family_clusters

    def compare_dilution_rates(self, year, province, degree_limits=[1, 2, 3, 4]):
        """
        Run the analysis for multiple degree limits and compare the results.
        This directly shows the "rate of dilution" as the ban scope widens.
        """
        comparison_results = []
        for limit in degree_limits:
            _, stats, _ = self.analyze_dynastic_power(year, province, limit)
            if stats:
                comparison_results.append(stats)

        results_df = pd.DataFrame(comparison_results)

        # Calculate the marginal dilution (change between limits)
        if len(results_df) > 1:
            results_df['marginal_seat_dilution'] = results_df['pct_seats_affected'].diff().fillna(results_df['pct_seats_affected'].iloc[0])
            results_df['marginal_hhi_change'] = results_df['political_hhi'].diff().fillna(results_df['political_hhi'].iloc[0])

        return results_df

    def plot_dilution_comparison(self, results_df):
        """Visualize the comparative dilution across degree limits."""
        if results_df.empty:
            print("No data to plot.")
            return
        if 'degree_limit' not in results_df.columns:
            print("Column 'degree_limit' missing. Available columns:", results_df.columns.tolist())
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Percentage of Seats Affected
        ax1.plot(results_df['degree_limit'], results_df['pct_seats_affected'], 
                 marker='o', linestyle='-', linewidth=2, markersize=8)
        ax1.set_xlabel('Consanguinity Degree Limit')
        ax1.set_ylabel('% of Seats Affected')
        ax1.set_title('Dynastic Dilution: Seats Opened Up')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(results_df['degree_limit'])

        # Plot 2: Political HHI (Power Concentration)
        ax2.plot(results_df['degree_limit'], results_df['political_hhi'], 
                 marker='s', linestyle='-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Consanguinity Degree Limit')
        ax2.set_ylabel('Political HHI (Power Concentration)')
        ax2.set_title('Power Concentration Under Different Bans')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(results_df['degree_limit'])

        plt.tight_layout()
        plt.show()

        # Print the status quo (0) for reference
        print(f"\n--- Analysis Summary ---")
        print(f"Status Quo (No Ban) - Seats affected: 0%, HHI: [Calculate from actual data]")

# -------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # Initialize simulator with your data files
    sim = DynastyDilutionSimulator("candidates.csv", "relationships.csv")

    # Run comparison for a specific province and year
    province = "Nueva Ecija"
    year = 2022

    # Compare degree limits 1 through 4 (and implicitly, status quo)
    results = sim.compare_dilution_rates(year, province, degree_limits=[1, 2, 3, 4])

    print(f"\nDynastic Dilution Comparison for {province} ({year})")
    print("="*60)
    print(results.to_string(index=False))

    # Visualize the results
    sim.plot_dilution_comparison(results)

    # Key Insight: The difference between the results for degree 2 and degree 4
    # shows the "rate of dilution" - how many more seats are affected by expanding the ban.
    if len(results) >= 4:
        print(f"\n--- Key Comparative Insights ---")
        print(f"Seats affected at 2nd degree: {results.iloc[1]['pct_seats_affected']}%")
        print(f"Seats affected at 4th degree: {results.iloc[3]['pct_seats_affected']}%")
        print(f"Additional dilution from 2nd to 4th degree: {results.iloc[3]['pct_seats_affected'] - results.iloc[1]['pct_seats_affected']}%")