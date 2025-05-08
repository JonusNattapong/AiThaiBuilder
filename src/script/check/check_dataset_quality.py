import pandas as pd
import numpy as np
from collections import Counter
import re
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.corpus import thai_words
import os
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

# นำเข้าฟังก์ชันวิเคราะห์ขั้นสูง
try:
    from advanced_quality_check import run_advanced_quality_check
    ADVANCED_CHECKS_AVAILABLE = True
except ImportError:
    ADVANCED_CHECKS_AVAILABLE = False

def setup_output_directories(base_dir):
    """Setup directory structure for analysis outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    
    # Create subdirectories
    subdirs = {
        'plots': os.path.join(output_dir, 'plots'),
        'reports': os.path.join(output_dir, 'reports'),
        'stats': os.path.join(output_dir, 'stats')
    }
    
    # Create all directories
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return subdirs, output_dir

def save_plot(fig, filename, plots_dir):
    """Save plot to file"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved plot: {filepath}")
    return filepath

def save_json(data, filename, stats_dir):
    """Save data as JSON"""
    filepath = os.path.join(stats_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved stats: {filepath}")
    return filepath

def save_report(text, filename, reports_dir):
    """Save text report"""
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved report: {filepath}")
    return filepath

def analyze_distribution(df, plots_dir):
    """Analyze label distribution"""
    print("Analyzing label distribution...")
    dist = df['label'].value_counts()
    percentages = (dist / len(df) * 100).round(2)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = save_plot(plt.gcf(), 'label_distribution.png', plots_dir)
    
    return {
        'distribution': {
            str(label): {
                'count': int(count),
                'percentage': f"{percentages[label]}%"
            } for label, count in dist.items()
        },
        'plot': plot_path
    }

def analyze_text_stats(df, plots_dir):
    """Analyze text statistics"""
    print("Analyzing text statistics...")
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    
    stats = {
        'text_length': {
            'mean': f"{df['text_length'].mean():.1f}",
            'median': f"{df['text_length'].median():.1f}",
            'min': f"{df['text_length'].min():.1f}",
            'max': f"{df['text_length'].max():.1f}"
        },
        'word_count': {
            'mean': f"{df['word_count'].mean():.1f}",
            'median': f"{df['word_count'].median():.1f}",
            'min': f"{df['word_count'].min():.1f}",
            'max': f"{df['word_count'].max():.1f}"
        }
    }
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(data=df, x='text_length', bins=50, ax=ax1)
    ax1.set_title('Text Length Distribution')
    sns.histplot(data=df, x='word_count', bins=50, ax=ax2)
    ax2.set_title('Word Count Distribution')
    plt.tight_layout()
    
    plot_path = save_plot(fig, 'text_statistics.png', plots_dir)
    
    return stats, plot_path

def analyze_thai_features(df, plots_dir):
    """Analyze Thai language features"""
    print("Analyzing Thai language features...")
    
    # Part of Speech analysis
    pos_counts = Counter()
    for text in df['text']:
        pos_tags = pos_tag(word_tokenize(text))
        pos_counts.update([tag for word, tag in pos_tags])
    
    # Known words analysis
    known_words = set(thai_words())
    df['known_word_ratio'] = df['text'].apply(
        lambda x: len([w for w in word_tokenize(x) if w in known_words]) / len(word_tokenize(x))
    )
    
    # Create POS plot
    plt.figure(figsize=(12, 6))
    pos_df = pd.DataFrame.from_dict(pos_counts, orient='index', columns=['count'])
    pos_df = pos_df.sort_values('count', ascending=False).head(10)
    plt.bar(pos_df.index, pos_df['count'])
    plt.title('Top 10 Part of Speech Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    pos_plot = save_plot(plt.gcf(), 'pos_distribution.png', plots_dir)
    
    # Create known words plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='known_word_ratio', bins=20)
    plt.title('Known Words Ratio Distribution')
    plt.xlabel('Ratio of Known Words')
    plt.ylabel('Count')
    plt.tight_layout()
    words_plot = save_plot(plt.gcf(), 'known_words_ratio.png', plots_dir)
    
    return {
        'pos_distribution': dict(pos_counts.most_common(10)),
        'known_words_stats': {
            'average_ratio': f"{df['known_word_ratio'].mean():.2%}",
            'min_ratio': f"{df['known_word_ratio'].min():.2%}",
            'max_ratio': f"{df['known_word_ratio'].max():.2%}"
        },
        'plots': {
            'pos_distribution': pos_plot,
            'known_words_ratio': words_plot
        }
    }

def generate_pdf_report(results, text_report, output_path):
    """Generate comprehensive PDF report"""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # สร้าง story สำหรับใส่เนื้อหา
    story = []
    styles = getSampleStyleSheet()
    
    # เพิ่มหัวข้อ
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Data quality analysis report", title_style))
    story.append(Spacer(1, 20))
    
    # ข้อมูลพื้นฐาน
    basic_info = [
        ["Data files:", results['dataset_info']['filename']],
        ["number of samples:", str(results['dataset_info']['total_samples'])],
        ["Date of analysis:", results['dataset_info']['timestamp']]
    ]
    
    tbl = Table(basic_info, colWidths=[120, 300])
    tbl.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12)
    ]))
    story.append(tbl)
    story.append(Spacer(1, 20))
    
    # เพิ่มกราฟต่างๆ
    for analysis_name, analysis_data in results['analyses'].items():
        # หัวข้อส่วน
        story.append(Paragraph(analysis_name.replace('_', ' ').title(), styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # เพิ่มกราฟ
        if 'plot' in analysis_data:
            img = Image(analysis_data['plot'])
            img.drawHeight = 300
            img.drawWidth = 450
            story.append(img)
            story.append(Spacer(1, 12))
        elif 'plots' in analysis_data:
            for plot_path in analysis_data['plots'].values():
                img = Image(plot_path)
                img.drawHeight = 300
                img.drawWidth = 450
                story.append(img)
                story.append(Spacer(1, 12))
        
        # เพิ่มข้อมูลสถิติ
        if isinstance(analysis_data, dict):
            stats_data = []
            for key, value in analysis_data.items():
                if key not in ['plot', 'plots']:
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            stats_data.append([f"{key} - {subkey}:", str(subvalue)])
                    else:
                        stats_data.append([key + ":", str(value)])
            
            if stats_data:
                tbl = Table(stats_data, colWidths=[200, 220])
                tbl.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12)
                ]))
                story.append(tbl)
                story.append(Spacer(1, 12))
    
    # สร้าง PDF
    doc.build(story)
    print(f"PDF report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Check dataset quality')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                      help='Base directory for analysis outputs')
    parser.add_argument('--advanced', action='store_true',
                      help='Run advanced quality checks (hallucination, bias, duplication)')
    parser.add_argument('--reference_file', type=str, default=None,
                      help='Path to reference data file for fact checking')
    parser.add_argument('--similarity_threshold', type=float, default=0.85,
                      help='Threshold for text similarity detection (0.0-1.0)')
    
    try:
        args = parser.parse_args()
        
        print(f"\nAnalyzing dataset: {args.input_file}")
        print("=" * 50)
        
        # Setup output directories
        print(f"\nSetting up output directories in {args.output_dir}...")
        dirs, output_dir = setup_output_directories(args.output_dir)
        
        # Load dataset
        print(f"\nLoading dataset...")
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            return
        
        df = pd.read_csv(args.input_file)
        print(f"Successfully loaded {len(df)} samples")
        
        # Run basic analyses
        results = {
            'dataset_info': {
                'filename': args.input_file,
                'total_samples': len(df),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'analyses': {}
        }
        
        # Label distribution
        dist_results = analyze_distribution(df, dirs['plots'])
        results['analyses']['label_distribution'] = dist_results
        
        # Text statistics
        stats_results, stats_plot = analyze_text_stats(df, dirs['plots'])
        results['analyses']['text_statistics'] = {
            'stats': stats_results,
            'plot': stats_plot
        }
        
        # Thai features
        thai_results = analyze_thai_features(df, dirs['plots'])
        results['analyses']['thai_features'] = thai_results
        
        # Save final results
        print("\nSaving analysis results...")
        save_json(results, 'analysis_results.json', dirs['stats'])
        
        # Generate and save report
        report = ["Dataset Quality Analysis Report", "=" * 30 + "\n"]
        report.append(f"Dataset: {args.input_file}")
        report.append(f"Total samples: {len(df)}")
        report.append(f"Analysis timestamp: {results['dataset_info']['timestamp']}\n")
        
        for section, data in results['analyses'].items():
            report.append(f"\n{section.upper()}")
            report.append("-" * len(section))
            if isinstance(data, dict):
                report.extend([f"{k}: {v}" for k, v in data.items() 
                             if k not in ['plot', 'plots']])
        
        # Generate PDF report
        print("\nGenerating PDF report...")
        pdf_path = os.path.join(output_dir, 'dataset_quality_report.pdf')
        generate_pdf_report(results, report, pdf_path)
        
        # Run advanced analyses if requested
        if args.advanced:
            if ADVANCED_CHECKS_AVAILABLE:
                print("\n=== Running Advanced Quality Checks ===")
                advanced_results = run_advanced_quality_check(
                    df, 
                    text_column='text',
                    output_dir=args.output_dir,
                    ref_data=None if not args.reference_file else args.reference_file,
                    similarity_threshold=args.similarity_threshold
                )
                
                # Add reference to advanced results in the basic report
                with open(os.path.join(dirs['reports'], 'basic_quality_report.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report))
                    f.write("\n\n=== ADVANCED ANALYSIS ===\n")
                    f.write(f"Advanced analysis results available at: {advanced_results['output_dir']}")
            else:
                print("\nWarning: Advanced quality checks were requested but the 'advanced_quality_check' module is not available.")
                print("Please make sure the module is installed correctly.")
        
        print("\nAnalysis complete!")
        print("=" * 50)
        print(f"Results saved in: {output_dir}")
        print(f"- Full PDF Report: {pdf_path}")
        print(f"- Individual files:")
        print(f"  - Plots: {dirs['plots']}")
        print(f"  - Report: {dirs['reports']}")
        print(f"  - Statistics: {dirs['stats']}")
        
    except Exception as e:
        print(f"\nError during analysis:")
        import traceback
        print(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
