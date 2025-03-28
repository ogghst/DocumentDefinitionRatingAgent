import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime
import matplotlib.patches as mpatches

# Set the matplotlib backend to Agg for saving figures
matplotlib.use('Agg')

class ChecklistReportGenerator:
    """Generates a PDF report from checklist analysis results."""
    
    def __init__(self, json_file_path, output_pdf_path=None):
        """Initialize the report generator.
        
        Args:
            json_file_path: Path to the JSON results file
            output_pdf_path: Path for the output PDF file. If None, will be derived from JSON filename.
        """
        self.json_file_path = json_file_path
        
        # Set default output path if not provided
        if output_pdf_path is None:
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            self.output_pdf_path = f"output/{base_name}_report.pdf"
        else:
            self.output_pdf_path = output_pdf_path
            
        # Load data
        try:
            with open(json_file_path, 'r') as f:
                self.results = json.load(f)
                
            # Check if results contain error messages
            if len(self.results) == 1 and ('error' in self.results[0] or 'message' in self.results[0] or 'warning' in self.results[0]):
                self.error_message = self.results[0].get('error') or self.results[0].get('message') or self.results[0].get('warning')
                self.has_valid_data = False
            else:
                self.error_message = None
                self.has_valid_data = True
                
        except Exception as e:
            self.results = []
            self.error_message = f"Error loading JSON data: {str(e)}"
            self.has_valid_data = False
            
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=24
        )
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=16
        )
        self.normal_style = self.styles['Normal']
        self.normal_centered = ParagraphStyle(
            'NormalCentered', 
            parent=self.styles['Normal'],
            alignment=TA_CENTER
        )
        
        # Extract phase information if available
        self.phase = "Unknown"
        if self.has_valid_data and len(self.results) > 0:
            if 'check_item' in self.results[0] and 'phase' in self.results[0]['check_item']:
                self.phase = self.results[0]['check_item']['phase']
    
    def create_summary_table(self):
        """Create a summary table of the check results."""
        if not self.has_valid_data:
            return None
            
        # Define the table data
        data = [['ID', 'Check Name', 'Status', 'Reliability', 'Needs Review']]
        
        for result in self.results:
            if 'check_item' not in result:
                continue
                
            check_id = result['check_item']['id']
            check_name = result['check_item']['name']
            status = "✓ MET" if result.get('is_met', False) else "✗ NOT MET"
            reliability = f"{result.get('reliability', 0):.1f}%"
            needs_review = "YES" if result.get('needs_human_review', False) else "NO"
            
            data.append([check_id, check_name, status, reliability, needs_review])
            
        # Create the table
        if len(data) > 1:  # Only if we have actual results
            table = Table(data, colWidths=[0.5*inch, 2*inch, 1*inch, 1*inch, 1*inch])
            
            # Style the table
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                ('ALIGN', (2, 1), (4, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ])
            
            # Add row colors for status
            for i, row in enumerate(data[1:], 1):
                # Get the status value from the row
                status = row[2]
                if "MET" in status:
                    style.add('BACKGROUND', (2, i), (2, i), colors.lightgreen)
                else:
                    style.add('BACKGROUND', (2, i), (2, i), colors.lightsalmon)
                    
                # Color for reliability
                reliability_val = float(row[3].strip('%'))
                if reliability_val >= 90:
                    style.add('BACKGROUND', (3, i), (3, i), colors.lightgreen)
                elif reliability_val >= 70:
                    style.add('BACKGROUND', (3, i), (3, i), colors.lightblue)
                elif reliability_val >= 50:
                    style.add('BACKGROUND', (3, i), (3, i), colors.lightyellow)
                else:
                    style.add('BACKGROUND', (3, i), (3, i), colors.lightsalmon)
                    
                # Color for needs review
                if row[4] == "YES":
                    style.add('BACKGROUND', (4, i), (4, i), colors.lightsalmon)
                else:
                    style.add('BACKGROUND', (4, i), (4, i), colors.lightgreen)
                    
            table.setStyle(style)
            return table
        
        return None
        
    def create_charts(self):
        """Create charts for the report."""
        if not self.has_valid_data:
            return []
            
        charts = []
        
        # 1. Status distribution pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        met_count = sum(1 for result in self.results if result.get('is_met', False))
        not_met_count = sum(1 for result in self.results if not result.get('is_met', False) and 'is_met' in result)
        
        if met_count > 0 or not_met_count > 0:
            labels = ['Met', 'Not Met']
            sizes = [met_count, not_met_count]
            colors = ['#90EE90', '#FFA07A']  # Light green and light salmon
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Checklist Items Status Distribution')
            
            # Save the figure to a BytesIO object
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight')
            img_data.seek(0)
            
            # Create an Image object
            img = Image(img_data, width=4*inch, height=3*inch)
            charts.append(img)
            plt.close(fig)
            
        # 2. Reliability distribution chart
        fig, ax = plt.subplots(figsize=(6, 4))
        reliability_scores = [result.get('reliability', 0) for result in self.results if 'reliability' in result]
        
        if reliability_scores:
            # Define reliability categories
            categories = ["90-100", "70-89", "50-69", "0-49"]
            counts = [
                sum(1 for score in reliability_scores if score >= 90),
                sum(1 for score in reliability_scores if 70 <= score < 90),
                sum(1 for score in reliability_scores if 50 <= score < 70),
                sum(1 for score in reliability_scores if score < 50)
            ]
            
            # Generate the bar chart
            colors = ['#90EE90', '#ADD8E6', '#FFFFE0', '#FFA07A']  # Green, blue, yellow, salmon
            ax.bar(categories, counts, color=colors)
            ax.set_xlabel('Reliability Range (%)')
            ax.set_ylabel('Number of Checks')
            ax.set_title('Distribution of Reliability Scores')
            
            # Add labels on top of bars
            for i, count in enumerate(counts):
                if count > 0:
                    ax.text(i, count + 0.1, str(count), ha='center')
            
            # Save the figure to a BytesIO object
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight')
            img_data.seek(0)
            
            # Create an Image object
            img = Image(img_data, width=4*inch, height=3*inch)
            charts.append(img)
            plt.close(fig)
            
        # 3. Branch distribution chart
        branch_data = {}
        for result in self.results:
            if 'check_item' in result and 'branch_name' in result['check_item']:
                branch = result['check_item']['branch_name']
                is_met = result.get('is_met', False)
                
                if branch not in branch_data:
                    branch_data[branch] = {'met': 0, 'not_met': 0}
                
                if is_met:
                    branch_data[branch]['met'] += 1
                else:
                    branch_data[branch]['not_met'] += 1
        
        if branch_data:
            fig, ax = plt.subplots(figsize=(7, 4))
            branches = list(branch_data.keys())
            met_counts = [branch_data[branch]['met'] for branch in branches]
            not_met_counts = [branch_data[branch]['not_met'] for branch in branches]
            
            # Create stacked bar chart
            x = range(len(branches))
            ax.bar(x, met_counts, label='Met', color='#90EE90')
            ax.bar(x, not_met_counts, bottom=met_counts, label='Not Met', color='#FFA07A')
            
            ax.set_xticks(x)
            ax.set_xticklabels(branches, rotation=45, ha='right')
            ax.set_ylabel('Number of Checks')
            ax.set_title('Status by Branch/Topic')
            ax.legend()
            
            plt.tight_layout()
            
            # Save the figure to a BytesIO object
            img_data = BytesIO()
            fig.savefig(img_data, format='png', bbox_inches='tight')
            img_data.seek(0)
            
            # Create an Image object
            img = Image(img_data, width=5*inch, height=3.5*inch)
            charts.append(img)
            plt.close(fig)
            
        return charts
    
    def create_detailed_sections(self):
        """Create detailed sections for each check result."""
        if not self.has_valid_data:
            return []
            
        elements = []
        
        for result in self.results:
            if 'check_item' not in result:
                continue
                
            check_item = result['check_item']
            
            # Create check header with colored background
            is_met = result.get('is_met', False)
            status_color = colors.lightgreen if is_met else colors.lightsalmon
            status_text = "✓ MET" if is_met else "✗ NOT MET"
            
            check_header = f"{check_item['id']}: {check_item['name']} ({status_text})"
            elements.append(Paragraph(check_header, self.subheading_style))
            
            # Check description
            elements.append(Paragraph(f"<b>Description:</b> {check_item['description']}", self.normal_style))
            
            # Branch/Topic
            elements.append(Paragraph(f"<b>Branch/Topic:</b> {check_item['branch_name']} (ID: {check_item['branch_id']})", self.normal_style))
            
            # Reliability score
            reliability = result.get('reliability', 0)
            reliability_text = f"<b>Reliability:</b> {reliability:.1f}%"
            elements.append(Paragraph(reliability_text, self.normal_style))
            
            # Needs human review
            needs_review = result.get('needs_human_review', False)
            review_text = f"<b>Needs Human Review:</b> {'Yes' if needs_review else 'No'}"
            elements.append(Paragraph(review_text, self.normal_style))
            
            # Analysis details
            if 'analysis_details' in result and result['analysis_details']:
                elements.append(Paragraph("<b>Analysis Details:</b>", self.normal_style))
                elements.append(Paragraph(result['analysis_details'], self.normal_style))
            
            # Sources
            if 'sources' in result and result['sources']:
                elements.append(Paragraph("<b>Evidence Sources:</b>", self.normal_style))
                for i, source in enumerate(result['sources'], 1):
                    elements.append(Paragraph(f"{i}. \"{source}\"", self.normal_style))
            
            # Add some space
            elements.append(Spacer(1, 0.25*inch))
            
        return elements
    
    def generate_pdf(self):
        """Generate the PDF report."""
        doc = SimpleDocTemplate(self.output_pdf_path, pagesize=A4)
        elements = []
        
        # Title
        title = f"Checklist Compliance Report - {self.phase}"
        elements.append(Paragraph(title, self.title_style))
        
        # Report metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {timestamp}", self.normal_centered))
        elements.append(Spacer(1, 0.5*inch))
        
        if not self.has_valid_data:
            # Error message
            elements.append(Paragraph("Error Report", self.heading_style))
            elements.append(Paragraph(self.error_message, self.normal_style))
        else:
            # Executive Summary
            elements.append(Paragraph("Executive Summary", self.heading_style))
            
            # Summary statistics
            total_checks = len(self.results)
            met_count = sum(1 for result in self.results if result.get('is_met', False))
            not_met_count = total_checks - met_count
            meets_percentage = (met_count / total_checks * 100) if total_checks > 0 else 0
            
            summary = f"""
            This report analyzes {total_checks} checklist items for the <b>{self.phase}</b> phase.
            <br/><br/>
            <b>Key Findings:</b>
            <br/>• {met_count} items ({meets_percentage:.1f}%) are met
            <br/>• {not_met_count} items ({100-meets_percentage:.1f}%) are not met
            <br/>• {sum(1 for result in self.results if result.get('needs_human_review', False))} items require human review
            """
            elements.append(Paragraph(summary, self.normal_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Summary Table
            elements.append(Paragraph("Results Summary", self.subheading_style))
            summary_table = self.create_summary_table()
            if summary_table:
                elements.append(summary_table)
            else:
                elements.append(Paragraph("No data available for summary table.", self.normal_style))
                
            elements.append(Spacer(1, 0.5*inch))
            
            # Charts and Visualizations
            elements.append(Paragraph("Charts and Visualizations", self.heading_style))
            charts = self.create_charts()
            
            for chart in charts:
                elements.append(chart)
                elements.append(Spacer(1, 0.2*inch))
                
            if not charts:
                elements.append(Paragraph("No data available for charts.", self.normal_style))
                
            # Detailed Results
            elements.append(Paragraph("Detailed Analysis", self.heading_style))
            detailed_sections = self.create_detailed_sections()
            elements.extend(detailed_sections)
            
        # Build the document
        doc.build(elements)
        print(f"PDF report generated: {self.output_pdf_path}")
        return self.output_pdf_path

if __name__ == "__main__":
    # Get the latest results file
    json_files = [f for f in os.listdir('.') if f.startswith('analysis_results_') and f.endswith('.json')]
    if not json_files:
        print("No analysis results files found!")
        exit(1)
        
    # Sort by modification time to get the most recent
    latest_file = max(json_files, key=lambda f: os.path.getmtime(f))
    print(f"Generating report from file: {latest_file}")
    
    # Generate the report
    generator = ChecklistReportGenerator(latest_file)
    output_path = generator.generate_pdf()
    
    print(f"Report generated successfully: {output_path}") 