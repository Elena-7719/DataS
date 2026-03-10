import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Настройка стиля
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 6)

# ------------------------------------------------------------
# ГИСТОГРАММЫ И РАСПРЕДЕЛЕНИЯ
# ------------------------------------------------------------
def plot_histograms(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df['total_bill'], bins=30, kde=True, ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Распределение суммы счета (total_bill)', fontsize=14)
    axes[0,0].set_xlabel('Сумма счета ($)')

    sns.histplot(df['tip'], bins=30, kde=True, ax=axes[0,1], color='salmon')
    axes[0,1].set_title('Распределение чаевых (tip)', fontsize=14)
    axes[0,1].set_xlabel('Чаевые ($)')

    sns.countplot(x='size', data=df, ax=axes[1,0], palette='pastel', hue='size', legend=False)
    axes[1,0].set_title('Размер компании (size)', fontsize=14)
    axes[1,0].set_xlabel('Количество человек')

    sns.scatterplot(x='total_bill', y='tip', data=df, ax=axes[1,1], alpha=0.6, color='purple')
    axes[1,1].set_title('Зависимость чаевых от суммы счета', fontsize=14)
    axes[1,1].set_xlabel('Сумма счета ($)')
    axes[1,1].set_ylabel('Чаевые ($)')

    plt.tight_layout()
    plt.suptitle('Общий обзор числовых признаков', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# ДИАГРАММЫ РАССЕЯНИЯ С ЦВЕТОВЫМ КОДИРОВАНИЕМ
# ------------------------------------------------------------
def plot_scatter_by_category(df, x='total_bill', y='tip', hue='sex'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=hue, data=df, alpha=0.7, s=80)
    plt.title(f'Зависимость {y} от {x} (по признаку {hue})', fontsize=15)
    plt.xlabel(x.replace('_', ' ').title())
    plt.ylabel(y.replace('_', ' ').title())
    plt.legend(title=hue.title())
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_all_scatters(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.scatterplot(x='total_bill', y='tip', hue='sex', data=df, ax=axes[0,0], alpha=0.7)
    axes[0,0].set_title('По полу')

    sns.scatterplot(x='total_bill', y='tip', hue='smoker', data=df, ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title('По наличию курящих')

    sns.scatterplot(x='total_bill', y='tip', hue='day', data=df, ax=axes[1,0], alpha=0.7)
    axes[1,0].set_title('По дням недели')

    sns.scatterplot(x='total_bill', y='tip', hue='time', data=df, ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('По времени (обед/ужин)')

    plt.tight_layout()
    plt.suptitle('Диаграммы рассеяния total_bill vs tip', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# ЯЩИКИ С УСАМИ (BOXPLOT) И VIOLINPLOT
# ------------------------------------------------------------
def plot_boxplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.boxplot(x='sex', y='tip', data=df, ax=axes[0,0], palette='pastel', hue='sex', legend=False)
    axes[0,0].set_title('Чаевые по полу')

    sns.boxplot(x='smoker', y='tip', data=df, ax=axes[0,1], palette='pastel', hue='smoker', legend=False)
    axes[0,1].set_title('Чаевые по наличию курящих')

    sns.boxplot(x='day', y='tip', data=df, ax=axes[1,0], palette='pastel', hue='day', legend=False)
    axes[1,0].set_title('Чаевые по дням')

    sns.boxplot(x='time', y='tip', data=df, ax=axes[1,1], palette='pastel', hue='time', legend=False)
    axes[1,1].set_title('Чаевые по времени')

    plt.tight_layout()
    plt.suptitle('Распределение чаевых по категориям', y=1.02, fontsize=16)
    plt.show()

def plot_violinplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.violinplot(x='sex', y='tip', data=df, ax=axes[0,0], palette='pastel', hue='sex', legend=False)
    axes[0,0].set_title('Чаевые по полу')

    sns.violinplot(x='smoker', y='tip', data=df, ax=axes[0,1], palette='pastel', hue='smoker', legend=False)
    axes[0,1].set_title('Чаевые по курению')

    sns.violinplot(x='day', y='tip', data=df, ax=axes[1,0], palette='pastel', hue='day', legend=False)
    axes[1,0].set_title('Чаевые по дням')

    sns.violinplot(x='time', y='tip', data=df, ax=axes[1,1], palette='pastel', hue='time', legend=False)
    axes[1,1].set_title('Чаевые по времени')

    plt.tight_layout()
    plt.suptitle('Violin plots чаевых', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# ЛИНЕЙНЫЕ ГРАФИКИ (СРЕДНИЕ ЗНАЧЕНИЯ)
# ------------------------------------------------------------
def plot_line_means(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    day_means = df.groupby('day')['tip'].mean().reindex(['Thur', 'Fri', 'Sat', 'Sun'])
    axes[0,0].plot(day_means.index, day_means.values, marker='o', linewidth=2, markersize=8, color='green')
    axes[0,0].set_title('Средние чаевые по дням')
    axes[0,0].set_ylabel('Средние чаевые ($)')
    axes[0,0].grid(True, alpha=0.3)

    time_means = df.groupby('time')['tip'].mean()
    axes[0,1].bar(time_means.index, time_means.values, color=['lightblue', 'lightcoral'])
    axes[0,1].set_title('Средние чаевые по времени')
    axes[0,1].set_ylabel('Средние чаевые ($)')

    size_means = df.groupby('size')['tip'].mean()
    axes[1,0].plot(size_means.index, size_means.values, marker='s', linewidth=2, markersize=8, color='purple')
    axes[1,0].set_title('Средние чаевые по размеру компании')
    axes[1,0].set_xlabel('Размер компании')
    axes[1,0].set_ylabel('Средние чаевые ($)')
    axes[1,0].grid(True, alpha=0.3)

    pivot = df.groupby(['sex', 'smoker'])['tip'].mean().unstack()
    pivot.plot(kind='bar', ax=axes[1,1], color=['pink', 'lightblue'])
    axes[1,1].set_title('Средние чаевые: пол × курение')
    axes[1,1].set_ylabel('Средние чаевые ($)')
    axes[1,1].legend(title='Smoker')

    plt.tight_layout()
    plt.suptitle('Средние значения чаевых', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# COUNT PLOTS ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ------------------------------------------------------------
def plot_countplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.countplot(x='sex', data=df, ax=axes[0,0], palette='pastel', hue='sex', legend=False)
    axes[0,0].set_title('Распределение по полу')

    sns.countplot(x='smoker', data=df, ax=axes[0,1], palette='pastel', hue='smoker', legend=False)
    axes[0,1].set_title('Распределение по курению')

    sns.countplot(x='day', data=df, ax=axes[1,0], palette='pastel', hue='day', legend=False)
    axes[1,0].set_title('Распределение по дням')

    sns.countplot(x='time', data=df, ax=axes[1,1], palette='pastel', hue='time', legend=False)
    axes[1,1].set_title('Распределение по времени')

    plt.tight_layout()
    plt.suptitle('Количество наблюдений по категориям', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# КОРРЕЛЯЦИОННАЯ МАТРИЦА
# ------------------------------------------------------------
def plot_correlation_matrix(df):
    numeric_df = df[['total_bill', 'tip', 'size']]
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=1, mask=mask, cbar_kws={"shrink": 0.8})
    plt.title('Корреляционная матрица числовых признаков', fontsize=15)
    plt.show()

# ------------------------------------------------------------
# ФАСЕТНЫЕ ГРАФИКИ (FacetGrid)
# ------------------------------------------------------------
def plot_facet_scatter(df):
    g = sns.FacetGrid(df, col='day', row='time', margin_titles=True, height=4)
    g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.6)
    g.add_legend()
    g.set_axis_labels('Сумма счета ($)', 'Чаевые ($)')
    g.fig.suptitle('Диаграммы рассеяния по дням и времени', y=1.02, fontsize=16)
    plt.show()

# ------------------------------------------------------------
# ФУНКЦИЯ ДЛЯ ЗАПУСКА ВСЕХ ГРАФИКОВ
# ------------------------------------------------------------
def plot_all_tips(df):
    print("Построение всех графиков для датасета Tips...")
    plot_histograms(df)
    plot_all_scatters(df)
    plot_boxplots(df)
    plot_violinplots(df)
    plot_line_means(df)
    plot_countplots(df)
    plot_correlation_matrix(df)
    plot_facet_scatter(df)
    print("✅ Все графики построены!")

