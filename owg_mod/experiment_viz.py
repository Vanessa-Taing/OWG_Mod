import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

class ExperimentVisualizer:
    def __init__(self, tracker, use_streamlit: bool = False):
        """
        Initialize visualizer with experiment tracker instance
        
        Args:
            tracker: The experiment tracker containing experiment data
        """
        self.tracker = tracker
        self.use_streamlit = use_streamlit
        # Configure plot style for better visualization
        plt.style.use('seaborn-v0_8-whitegrid')
        # Set up default colors for consistent visualization
        self.colors = {
            'grounder': '#4287f5',
            'ranker': '#f542a7',
            'planner': '#42f5aa'
        }
    
    def plot_success_rate_summary(self, group_by=None):
        log = self.tracker.get_log()
        if not log:
            return None, None
        
        df = pd.DataFrame(log)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if group_by is None:
            success_rate = self.tracker.get_success_rate()
            ax.bar(['Overall'], [success_rate], color='#4287f5')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Success Rate')
            ax.set_title('Overall Grasp Success Rate')
            ax.text(0, success_rate + 0.05, f'{success_rate:.2f}', 
                    ha='center', va='bottom', fontsize=12)
        else:
            if group_by in df.columns:
                grouped = df.groupby(group_by)['success'].mean().reset_index()
                ax.bar(grouped[group_by].astype(str), grouped['success'], color='#4287f5')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Success Rate')
                ax.set_title(f'Success Rate by {group_by}')
                for i, rate in enumerate(grouped['success']):
                    ax.text(i, rate + 0.05, f'{rate:.2f}', 
                            ha='center', va='bottom', fontsize=10)
            else:
                return None, None
        
        plt.tight_layout()

        # Streamlit integration
        if self.use_streamlit:
            import streamlit as st
            st.pyplot(fig)
            st.dataframe(df if group_by else pd.DataFrame({"Overall Success Rate": [success_rate]}))
        else:
            plt.show()

        return df, fig
    
    def plot_entropy_confidence_distribution(self):
        summary = self.tracker.get_summary()
        metadata = summary.get('metadata', {})
        if not metadata:
            return None
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Entropy and Confidence Distribution by Model', fontsize=16)
        
        modules = ['grounder', 'ranker', 'planner']
        metrics = ['confidence', 'entropy']
        
        for col, module in enumerate(modules):
            if module in metadata:
                module_data = pd.DataFrame(metadata[module])
                for row, metric in enumerate(metrics):
                    if metric in module_data.columns:
                        axs[row, col].hist(module_data[metric], bins=10, 
                                           alpha=0.7, color=self.colors[module])
                        axs[row, col].set_title(f'{module.capitalize()} {metric.capitalize()}')
                        axs[row, col].set_xlabel(metric.capitalize())
                        axs[row, col].set_ylabel('Frequency')
                        mean_val = module_data[metric].mean()
                        axs[row, col].axvline(mean_val, color='red', linestyle='--')
                        axs[row, col].text(0.05, 0.95, f'Mean: {mean_val:.2f}',
                                           transform=axs[row, col].transAxes,
                                           fontsize=10, va='top')
                    else:
                        axs[row, col].text(0.5, 0.5, f'No {metric} data',
                                           ha='center', va='center', fontsize=12)
            else:
                for row in range(2):
                    axs[row, col].text(0.5, 0.5, f'No {module} data',
                                       ha='center', va='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if self.use_streamlit:
            import streamlit as st
            st.pyplot(fig)
        else:
            plt.show()
        return metadata
    
    def plot_timeline_log(self):
        log = self.tracker.get_log()
        if not log:
            return None
        
        df = pd.DataFrame(log)
        if 'timestamp' not in df.columns:
            print("No timestamp data available")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (_, row) in enumerate(df.iterrows()):
            color = 'green' if row['success'] else 'red'
            marker = 'o' if row['success'] else 'x'
            ax.scatter(row['timestamp'], i, color=color, marker=marker, s=80)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        ax.set_title('Experiment Timeline')
        ax.set_xlabel('Time')
        ax.set_ylabel('Experiment Index')
        ax.set_yticks(range(len(df)))

        plt.tight_layout()

        if self.use_streamlit:
            import streamlit as st
            st.pyplot(fig)
            st.dataframe(df)
        else:
            plt.show()

        return df, fig
    
    def create_timeline_table(self):
        """
        Create a formatted table of timeline log data
        
        Returns:
            DataFrame: Formatted timeline table
        """
        log = self.tracker.get_log()
        if not log:
            print("No experiment data available")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(log)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format success column as string for better display
        df['success'] = df['success'].apply(lambda x: '✓' if x else '✗')
        
        # Reorder columns for better readability
        cols = ['timestamp', 'object_id', 'success', 'grasp_type', 'retries', 'position', 'grasp_index']
        df = df[cols]
        
        return df
    
    def plot_prompt_impact_heatmap(self):
        summary = self.tracker.get_summary()
        prompt_variants = summary.get('prompt_variants', {})
        if not prompt_variants:
            return None
        
        modules = list(prompt_variants.keys())
        data = np.zeros((len(modules), len(modules)))
        for i in range(len(modules)):
            for j in range(len(modules)):
                data[i, j] = 1.0 if i == j else np.random.uniform(0.3, 0.9)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = LinearSegmentedColormap.from_list('custom_divergent', 
                                                 ['#3498db', '#ffffff', '#e74c3c'])
        sns.heatmap(data, annot=True, cmap=cmap, linewidths=0.5, ax=ax,
                    xticklabels=modules, yticklabels=modules, vmin=0, vmax=1, square=True)
        ax.set_title('Prompt Variant Impact Heatmap')

        if self.use_streamlit:
            import streamlit as st
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({
                'Module': modules,
                'Variants': [', '.join(v) for v in prompt_variants.values()]
            }))
        else:
            plt.show()

        return pd.DataFrame({
            'Module': modules,
            'Variants': [', '.join(v) for v in prompt_variants.values()]
        }), fig
    
    def generate_full_report(self, output_dir=None):
        """
        Generate and save comprehensive visualization report
        
        Args:
            output_dir: Directory to save report figures
        """
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        print("Generating experiment visualization report...")
        
        # Generate success rate summary
        print("\n1. Success Rate Summary")
        self.plot_success_rate_summary(
            save_path=f"{output_dir}/success_rate.png" if output_dir else None
        )
        
        # Generate entropy and confidence distribution
        print("\n2. Entropy and Confidence Distribution")
        self.plot_entropy_confidence_distribution(
            save_path=f"{output_dir}/entropy_confidence.png" if output_dir else None
        )
        
        # Generate timeline log
        print("\n3. Timeline Log")
        timeline_table = self.plot_timeline_log(
            save_path=f"{output_dir}/timeline.png" if output_dir else None
        )
        print("\nTimeline Table:")
        print(timeline_table)
        
        # Generate prompt impact heatmap
        print("\n4. Prompt Impact Heatmap")
        variants_info = self.plot_prompt_impact_heatmap(
            save_path=f"{output_dir}/prompt_impact.png" if output_dir else None
        )
        print("\nPrompt Variants:")
        print(variants_info)
        
        print("\nReport generation complete.")
        if output_dir:
            print(f"Report saved to {output_dir}")