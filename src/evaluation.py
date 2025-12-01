"""
Módulo de evaluación y comparación de modelos de reconocimiento de emociones.
Calcula métricas de rendimiento y genera visualizaciones comparativas.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime
import json
import time

from .config import EMOTIONS, RESULTS_DIR


class ModelEvaluator:
    """
    Evaluador de modelos de reconocimiento de emociones.
    Calcula métricas y genera visualizaciones comparativas.
    """

    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def evaluate_predictions(self, y_true, y_pred, model_name):
        """
        Evalúa las predicciones de un modelo.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            model_name: Nombre del modelo

        Returns:
            Diccionario con métricas
        """
        # Filtrar predicciones nulas
        valid_mask = [p is not None for p in y_pred]
        y_true_valid = [y for y, v in zip(y_true, valid_mask) if v]
        y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

        if not y_true_valid:
            return None

        # Calcular métricas generales
        metrics = {
            'model_name': model_name,
            'total_samples': len(y_true),
            'valid_predictions': len(y_true_valid),
            'failed_predictions': len(y_true) - len(y_true_valid),
            'accuracy': accuracy_score(y_true_valid, y_pred_valid),
            'precision_macro': precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
        }

        # Métricas por clase
        metrics['per_class'] = {}
        for emotion in EMOTIONS:
            emotion_mask = [y == emotion for y in y_true_valid]
            if sum(emotion_mask) > 0:
                y_true_emotion = [y for y, m in zip(y_true_valid, emotion_mask) if m]
                y_pred_emotion = [p for p, m in zip(y_pred_valid, emotion_mask) if m]

                correct = sum(1 for t, p in zip(y_true_emotion, y_pred_emotion) if t == p)
                total = len(y_true_emotion)

                metrics['per_class'][emotion] = {
                    'total': total,
                    'correct': correct,
                    'accuracy': correct / total if total > 0 else 0
                }

        # Matriz de confusión
        labels = sorted(list(set(y_true_valid + y_pred_valid)))
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['labels'] = labels

        # Reporte de clasificación
        metrics['classification_report'] = classification_report(
            y_true_valid, y_pred_valid, labels=EMOTIONS, zero_division=0, output_dict=True
        )

        self.results[model_name] = metrics
        return metrics

    def compare_models(self, model_results):
        """
        Compara los resultados de múltiples modelos.

        Args:
            model_results: Diccionario {nombre_modelo: métricas}

        Returns:
            DataFrame con comparación
        """
        comparison_data = []

        for model_name, metrics in model_results.items():
            row = {
                'Modelo': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1-Score (Weighted)': metrics['f1_weighted'],
                'Muestras Válidas': metrics['valid_predictions'],
                'Predicciones Fallidas': metrics['failed_predictions']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def plot_confusion_matrices(self, save_path=None):
        """
        Genera visualización de matrices de confusión para todos los modelos.

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        n_models = len(self.results)
        if n_models == 0:
            print("No hay resultados para visualizar")
            return

        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 7))

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, metrics) in zip(axes, self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            labels = metrics['labels']

            # Normalizar matriz de confusión
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )
            ax.set_title(f'Matriz de Confusión - {model_name}')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en {save_path}")

        plt.close()

    def plot_metrics_comparison(self, save_path=None):
        """
        Genera gráfico comparativo de métricas entre modelos.

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if len(self.results) == 0:
            print("No hay resultados para visualizar")
            return

        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        models = list(self.results.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.35 if len(models) == 2 else 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            values = [self.results[model][m] for m in metrics_to_plot]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model)

            # Añadir valores sobre las barras
            for bar, val in zip(bars, values):
                ax.annotate(
                    f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9
                )

        ax.set_xlabel('Métrica')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Métricas entre Modelos')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación de métricas guardada en {save_path}")

        plt.close()

    def plot_per_emotion_comparison(self, save_path=None):
        """
        Genera gráfico de accuracy por emoción para cada modelo.

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if len(self.results) == 0:
            print("No hay resultados para visualizar")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        models = list(self.results.keys())
        x = np.arange(len(EMOTIONS))
        width = 0.35 if len(models) == 2 else 0.25

        for i, model in enumerate(models):
            accuracies = []
            for emotion in EMOTIONS:
                if emotion in self.results[model]['per_class']:
                    acc = self.results[model]['per_class'][emotion]['accuracy']
                else:
                    acc = 0
                accuracies.append(acc)

            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, accuracies, width, label=model)

        ax.set_xlabel('Emoción')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy por Emoción y Modelo')
        ax.set_xticks(x)
        ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación por emoción guardada en {save_path}")

        plt.close()

    def plot_processing_time(self, times, save_path=None):
        """
        Genera gráfico de tiempos de procesamiento.

        Args:
            times: Diccionario {modelo: tiempo_en_segundos}
            save_path: Ruta para guardar la imagen (opcional)
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        models = list(times.keys())
        values = list(times.values())

        bars = ax.bar(models, values, color=['#3498db', '#e74c3c'])

        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.2f}s',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        ax.set_xlabel('Modelo')
        ax.set_ylabel('Tiempo (segundos)')
        ax.set_title('Tiempo de Procesamiento por Modelo')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tiempos de procesamiento guardados en {save_path}")

        plt.close()

    def generate_report(self, processing_times=None):
        """
        Genera un reporte completo de la evaluación.

        Args:
            processing_times: Diccionario con tiempos de procesamiento

        Returns:
            String con el reporte
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE EVALUACIÓN COMPARATIVA")
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        for model_name, metrics in self.results.items():
            report.append(f"\n{'='*40}")
            report.append(f"MODELO: {model_name}")
            report.append(f"{'='*40}")

            report.append(f"\nMuestras totales: {metrics['total_samples']}")
            report.append(f"Predicciones válidas: {metrics['valid_predictions']}")
            report.append(f"Predicciones fallidas: {metrics['failed_predictions']}")

            report.append(f"\nMÉTRICAS GENERALES:")
            report.append(f"  Accuracy:           {metrics['accuracy']:.4f}")
            report.append(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
            report.append(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
            report.append(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
            report.append(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
            report.append(f"  Recall (Weighted):    {metrics['recall_weighted']:.4f}")
            report.append(f"  F1-Score (Weighted):  {metrics['f1_weighted']:.4f}")

            report.append(f"\nRENDIMIENTO POR EMOCIÓN:")
            for emotion, data in metrics['per_class'].items():
                report.append(
                    f"  {emotion.capitalize():10s}: "
                    f"Acc={data['accuracy']:.4f} "
                    f"({data['correct']}/{data['total']})"
                )

            if processing_times and model_name in processing_times:
                report.append(f"\nTIEMPO DE PROCESAMIENTO: {processing_times[model_name]:.2f} segundos")

        # Comparación
        if len(self.results) > 1:
            report.append(f"\n{'='*80}")
            report.append("COMPARACIÓN DE MODELOS")
            report.append(f"{'='*80}")

            df = self.compare_models(self.results)
            report.append(f"\n{df.to_string(index=False)}")

            # Identificar mejor modelo por métrica
            report.append("\nMEJOR MODELO POR MÉTRICA:")
            metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

            for metric, name in zip(metrics_to_compare, metric_names):
                best_model = max(self.results.keys(), key=lambda m: self.results[m][metric])
                best_value = self.results[best_model][metric]
                report.append(f"  {name}: {best_model} ({best_value:.4f})")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_results(self, filename=None):
        """
        Guarda los resultados en archivos JSON y texto.

        Args:
            filename: Nombre base del archivo (sin extensión)
        """
        if filename is None:
            filename = f"evaluation_results_{self.timestamp}"

        # Guardar JSON
        json_path = os.path.join(RESULTS_DIR, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Resultados guardados en {json_path}")

        # Guardar reporte de texto
        txt_path = os.path.join(RESULTS_DIR, f"{filename}_report.txt")
        with open(txt_path, 'w') as f:
            f.write(self.generate_report())
        print(f"Reporte guardado en {txt_path}")

        return json_path, txt_path
