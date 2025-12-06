"""
Visualizations - Plotly Charts and Dashboards
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import List, Dict


def create_occupancy_bar_chart(ward_summary: List[Dict]) -> go.Figure:
    """Create bar chart for ward occupancy percentage"""
    fig_bar = go.Figure()
    for item in ward_summary:
        fig_bar.add_trace(go.Bar(
            x=[item['ward_type']],
            y=[item['occupancy_pct']],
            name=item['ward_type'],
            text=f"{item['occupancy_pct']}%",
            textposition='outside',
            marker=dict(
                color='red' if item['occupancy_pct'] >= 95 else
                      'orange' if item['occupancy_pct'] >= 85 else
                      'yellow' if item['occupancy_pct'] >= 70 else 'green'
            )
        ))
    
    fig_bar.update_layout(
        title="Ward Occupancy % by Type",
        yaxis_title="Occupancy %",
        xaxis_title="Ward Type",
        showlegend=False,
        height=400,
        hovermode='x unified'
    )
    return fig_bar


def create_bed_availability_chart(ward_summary: List[Dict]) -> go.Figure:
    """Create stacked bar chart for occupied vs available beds"""
    fig_stack = go.Figure()
    ward_types = [item['ward_type'] for item in ward_summary]
    occupied = [item['occupied_beds'] for item in ward_summary]
    available = [item['available_beds'] for item in ward_summary]
    
    fig_stack.add_trace(go.Bar(
        x=ward_types,
        y=occupied,
        name='Occupied Beds',
        marker_color='indianred'
    ))
    fig_stack.add_trace(go.Bar(
        x=ward_types,
        y=available,
        name='Available Beds',
        marker_color='lightgreen'
    ))
    
    fig_stack.update_layout(
        barmode='stack',
        title="Bed Availability by Ward Type",
        yaxis_title="Number of Beds",
        xaxis_title="Ward Type",
        height=400,
        hovermode='x unified'
    )
    return fig_stack


def create_grouped_occupancy_chart(breakdown_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart by hospital and ward"""
    fig_grouped = px.bar(
        breakdown_df,
        x='Ward',
        y='Occupancy %',
        color='Hospital',
        barmode='group',
        title="Occupancy % by Hospital & Ward",
        height=400
    )
    fig_grouped.update_yaxes(range=[0, 105])
    return fig_grouped


def create_occupancy_heatmap(breakdown_df: pd.DataFrame) -> go.Figure:
    """Create heatmap for hospital vs ward occupancy"""
    pivot_df = breakdown_df.pivot_table(
        index='Hospital',
        columns='Ward',
        values='Occupancy %',
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else None
    )
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdYlGn_r',
        text=pivot_df.values,
        texttemplate='%{text:.0f}%',
        textfont={"size": 12},
        colorbar=dict(title="Occupancy %")
    ))
    
    fig_heatmap.update_layout(
        title="Hospital & Ward Occupancy Heatmap",
        xaxis_title="Ward Type",
        yaxis_title="Hospital",
        height=400
    )
    return fig_heatmap


def create_ward_distribution_chart(ward_count: pd.Series) -> go.Figure:
    """Create bar chart for admissions by ward type"""
    fig_ward = px.bar(
        x=ward_count.index,
        y=ward_count.values,
        title="Number of Admissions by Ward Type",
        labels={'x': 'Ward Type', 'y': 'Number of Admissions'},
        color=ward_count.index,
        height=350,
        color_discrete_map={
            'General': '#1f77b4',
            'ICU': '#ff7f0e',
            'Neurological': '#2ca02c',
            'Outpatient': '#d62728'
        }
    )
    fig_ward.update_layout(showlegend=False)
    return fig_ward


def create_diagnosis_chart(diagnosis_count: pd.Series) -> go.Figure:
    """Create horizontal bar chart for top diagnoses"""
    fig_diagnosis = px.bar(
        x=diagnosis_count.values,
        y=diagnosis_count.index,
        orientation='h',
        title="Most Common Diagnoses",
        labels={'x': 'Count', 'y': 'Diagnosis Code'},
        height=350
    )
    return fig_diagnosis


def create_severity_pie_chart(severity_count: pd.Series) -> go.Figure:
    """Create pie chart for severity distribution"""
    fig_severity = px.pie(
        values=severity_count.values,
        names=severity_count.index,
        title="Patient Risk Distribution",
        height=350,
        color_discrete_sequence=['#2ca02c', '#ffd700', '#ff7f0e', '#d62728']
    )
    return fig_severity


def create_hospital_ward_heatmap(hospital_ward: pd.DataFrame) -> go.Figure:
    """Create heatmap for hospital-ward admission cross-tabulation"""
    fig_heatmap = px.imshow(
        hospital_ward,
        labels=dict(x='Ward Type', y='Hospital', color='Admissions'),
        title="Hospital-Ward Admission Heatmap",
        height=400,
        color_continuous_scale='YlOrRd'
    )
    return fig_heatmap
