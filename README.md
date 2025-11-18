
import numpy as np

import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Configuración de estilo de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

print("="*70)
print("SIMULACIONES COMPUTACIONALES - ECUACIONES DIFERENCIALES")
print("="*70)

# ============================================================================
# PROBLEMA 1: GRADIENTE DESCENDENTE (PRIMER ORDEN)
# ============================================================================

def gradient_descent_ode(theta, t, alpha, theta_star):
    """
    Ecuación diferencial: dθ/dt = -α(θ - θ*)
    
    Parámetros:
        theta: valor actual del parámetro
        t: tiempo
        alpha: tasa de aprendizaje
        theta_star: valor óptimo
    """
    return -alpha * (theta - theta_star)

def simular_gradiente_descendente():
    """Simula y visualiza el proceso de gradiente descendente"""
    
    print("\n[1/2] Ejecutando: GRADIENTE DESCENDENTE")
    print("-" * 70)
    
    # Parámetros del problema
    theta_star = 5.0      # Valor óptimo
    theta_0 = 0.0         # Valor inicial
    alphas = [0.01, 0.1, 0.5]  # Diferentes tasas de aprendizaje
    t = np.linspace(0, 50, 500)  # Tiempo de 0 a 50 iteraciones
    
    # Crear figura con 6 subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PROBLEMA 1: OPTIMIZACIÓN CON GRADIENTE DESCENDENTE', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ========== Subplot 1: Convergencia de θ(t) ==========
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, alpha in enumerate(alphas):
        theta = odeint(gradient_descent_ode, theta_0, t, args=(alpha, theta_star))
        ax1.plot(t, theta, label=f'α = {alpha}', linewidth=2.5, color=colors[i])
    
    ax1.axhline(y=theta_star, color='black', linestyle='--', 
                linewidth=2, label='θ* (óptimo)', alpha=0.7)
    ax1.set_xlabel('Tiempo (iteraciones)', fontweight='bold')
    ax1.set_ylabel('θ(t)', fontweight='bold')
    ax1.set_title('Convergencia según Tasa de Aprendizaje', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 6)
    
    # ========== Subplot 2: Error en escala logarítmica ==========
    ax2 = plt.subplot(2, 3, 2)
    
    for i, alpha in enumerate(alphas):
        theta = odeint(gradient_descent_ode, theta_0, t, args=(alpha, theta_star))
        error = np.abs(theta.flatten() - theta_star)
        ax2.semilogy(t, error, label=f'α = {alpha}', linewidth=2.5, color=colors[i])
    
    ax2.axhline(y=0.01*abs(theta_0-theta_star), color='red', 
                linestyle='--', linewidth=2, alpha=0.7, label='1% error inicial')
    ax2.set_xlabel('Tiempo (iteraciones)', fontweight='bold')
    ax2.set_ylabel('|Error| = |θ(t) - θ*|', fontweight='bold')
    ax2.set_title('Reducción Exponencial del Error', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # ========== Subplot 3: Validación analítica vs numérica ==========
    ax3 = plt.subplot(2, 3, 3)
    alpha = 0.1
    
    theta_num = odeint(gradient_descent_ode, theta_0, t, args=(alpha, theta_star))
    theta_ana = theta_star + (theta_0 - theta_star) * np.exp(-alpha * t)
    
    ax3.plot(t, theta_num, 'b-', label='Solución Numérica', linewidth=3)
    ax3.plot(t, theta_ana, 'r--', label='Solución Analítica', linewidth=2.5)
    ax3.set_xlabel('Tiempo (iteraciones)', fontweight='bold')
    ax3.set_ylabel('θ(t)', fontweight='bold')
    ax3.set_title(f'Validación del Modelo (α = {alpha})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Mostrar error numérico
    error_max = np.max(np.abs(theta_num.flatten() - theta_ana))
    ax3.text(0.05, 0.95, f'Error máximo: {error_max:.2e}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ========== Subplot 4: Velocidad de cambio dθ/dt ==========
    ax4 = plt.subplot(2, 3, 4)
    alpha = 0.1
    
    theta = odeint(gradient_descent_ode, theta_0, t, args=(alpha, theta_star))
    dtheta_dt = -alpha * (theta - theta_star)
    
    ax4.plot(t, dtheta_dt, 'g-', linewidth=2.5)
    ax4.fill_between(t, 0, dtheta_dt.flatten(), alpha=0.3, color='green')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Tiempo (iteraciones)', fontweight='bold')
    ax4.set_ylabel('dθ/dt', fontweight='bold')
    ax4.set_title('Velocidad de Cambio del Parámetro', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ========== Subplot 5: Función de costo J(θ) ==========
    ax5 = plt.subplot(2, 3, 5)
    
    theta_range = np.linspace(-2, 12, 300)
    J = 0.5 * (theta_range - theta_star)**2
    
    ax5.plot(theta_range, J, 'purple', linewidth=3, label='J(θ) = ½(θ-θ*)²')
    
    # Trayectorias
    for i, alpha in enumerate([0.01, 0.1, 0.5]):
        theta_traj = odeint(gradient_descent_ode, theta_0, t, args=(alpha, theta_star))
        J_traj = 0.5 * (theta_traj - theta_star)**2
        indices = np.linspace(0, len(t)-1, 10, dtype=int)
        ax5.plot(theta_traj[indices], J_traj[indices], 'o-', 
                 linewidth=2, alpha=0.7, label=f'α={alpha}', color=colors[i])
    
    ax5.plot(theta_star, 0, 'r*', markersize=20, label='Mínimo global')
    ax5.set_xlabel('θ', fontweight='bold')
    ax5.set_ylabel('J(θ)', fontweight='bold')
    ax5.set_title('Función de Costo y Trayectorias', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.5, 15)
    
    # ========== Subplot 6: Tabla de resultados ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    data = []
    for alpha in [0.001, 0.01, 0.1, 0.5]:
        tau = 1 / alpha
        t_99 = np.log(100) / alpha
        data.append([f'{alpha}', f'{tau:.1f}', f'{t_99:.1f}'])
    
    table = ax6.table(cellText=data,
                      colLabels=['α', 'τ (const. tiempo)', 't₉₉% (iters)'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.25, 0.35, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax6.text(0.5, 0.85, 'Análisis de Convergencia', 
             ha='center', fontsize=12, fontweight='bold', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig('simulacion_gradiente_descendente.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: simulacion_gradiente_descendente.png")
    
    # Análisis numérico
    print("\nRESULTADOS NUMÉRICOS:")
    for alpha in alphas:
        t_99 = np.log(100) / alpha
        print(f"  α = {alpha:5.2f} → Convergencia al 1% en {t_99:6.1f} iteraciones")
    
    return fig

# ============================================================================
# PROBLEMA 2: SISTEMA DE COLAS (ORDEN SUPERIOR)
# ============================================================================

def queue_system(y, t, xi, omega_0, lambda_0, A, omega):
    """
    Sistema de ecuaciones de primer orden para la cola de segundo orden
    
    y = [N, v] donde:
        N = número de usuarios en cola
        v = dN/dt (velocidad de cambio)
    
    Sistema:
        dN/dt = v
        dv/dt = -2ξω₀v - ω₀²N + λ₀ + A·cos(ωt)
    """
    N, v = y
    dN_dt = v
    dv_dt = -2*xi*omega_0*v - omega_0**2*N + lambda_0 + A*np.cos(omega*t)
    return [dN_dt, dv_dt]

def simular_sistema_colas():
    """Simula y visualiza el sistema de colas"""
    
    print("\n[2/2] Ejecutando: SISTEMA DE COLAS")
    print("-" * 70)
    
    # Parámetros del sistema
    omega_0 = 0.1        # Frecuencia natural (rad/min)
    lambda_0 = 50        # Demanda promedio (usuarios/min)
    A = 20               # Amplitud de variación
    omega = 2*np.pi/1440 # Ciclo de 24 horas
    
    # Diferentes amortiguamientos
    xis = [0.1, 0.3, 0.7, 1.0]
    xi_labels = ['ξ=0.1 (sub)', 'ξ=0.3 (sub)', 'ξ=0.7 (sub)', 'ξ=1.0 (crítico)']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # Condiciones iniciales
    N0 = 30
    v0 = 0
    y0 = [N0, v0]
    
    # Tiempo de simulación
    t = np.linspace(0, 480, 2000)  # 8 horas = 480 minutos
    
    # Crear figura
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('PROBLEMA 2: SISTEMA DE COLAS EN APLICACIÓN DE ATENCIÓN AL CLIENTE', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ========== Subplot 1: Evolución de N(t) ==========
    ax1 = plt.subplot(2, 3, 1)
    
    N_eq = lambda_0 / omega_0**2
    
    for i, (xi, label) in enumerate(zip(xis, xi_labels)):
        sol = odeint(queue_system, y0, t, args=(xi, omega_0, lambda_0, A, omega))
        N = sol[:, 0]
        ax1.plot(t, N, label=label, linewidth=2.5, color=colors[i])
    
    ax1.axhline(y=N_eq, color='red', linestyle='--', 
                linewidth=2, label=f'Equilibrio ({N_eq:.0f} usuarios)', alpha=0.7)
    ax1.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax1.set_ylabel('Usuarios en Cola', fontweight='bold')
    ax1.set_title('Dinámica según Amortiguamiento', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== Subplot 2: Zoom en transitorio ==========
    ax2 = plt.subplot(2, 3, 2)
    t_zoom = t[t <= 150]
    
    for i, (xi, label) in enumerate(zip(xis, xi_labels)):
        sol = odeint(queue_system, y0, t_zoom, args=(xi, omega_0, lambda_0, A, omega))
        N = sol[:, 0]
        ax2.plot(t_zoom, N, label=label, linewidth=2.5, color=colors[i])
    
    ax2.axhline(y=N_eq, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax2.set_ylabel('Usuarios en Cola', fontweight='bold')
    ax2.set_title('Fase Transitoria (0-150 min)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== Subplot 3: Velocidad dN/dt ==========
    ax3 = plt.subplot(2, 3, 3)
    xi = 0.3
    
    sol = odeint(queue_system, y0, t, args=(xi, omega_0, lambda_0, A, omega))
    v = sol[:, 1]
    
    ax3.plot(t, v, 'purple', linewidth=2.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.fill_between(t, 0, v, where=(v>=0), alpha=0.3, 
                     color='green', label='Cola aumenta')
    ax3.fill_between(t, 0, v, where=(v<0), alpha=0.3, 
                     color='red', label='Cola disminuye')
    ax3.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax3.set_ylabel('dN/dt (usuarios/min)', fontweight='bold')
    ax3.set_title(f'Tasa de Cambio (ξ = {xi})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== Subplot 4: Espacio de fases ==========
    ax4 = plt.subplot(2, 3, 4)
    
    for i, (xi, label) in enumerate(zip([0.1, 0.3, 1.0], 
                                        ['Sub (ξ=0.1)', 'Óptimo (ξ=0.3)', 'Crítico (ξ=1.0)'])):
        sol = odeint(queue_system, y0, t[:800], args=(xi, omega_0, lambda_0, 0, omega))
        N = sol[:, 0]
        v = sol[:, 1]
        ax4.plot(N, v, linewidth=2, label=label, alpha=0.8, color=colors[i])
        ax4.plot(N[0], v[0], 'o', markersize=10, color=colors[i])
    
    ax4.plot(N_eq, 0, 'r*', markersize=15, label='Punto de equilibrio')
    ax4.set_xlabel('N (usuarios)', fontweight='bold')
    ax4.set_ylabel('dN/dt (tasa de cambio)', fontweight='bold')
    ax4.set_title('Diagrama de Fase', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=N_eq, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    # ========== Subplot 5: Comparación con/sin variación ==========
    ax5 = plt.subplot(2, 3, 5)
    xi = 0.3
    
    sol_con = odeint(queue_system, y0, t, args=(xi, omega_0, lambda_0, A, omega))
    N_con = sol_con[:, 0]
    
    sol_sin = odeint(queue_system, y0, t, args=(xi, omega_0, lambda_0, 0, omega))
    N_sin = sol_sin[:, 0]
    
    ax5.plot(t, N_con, 'b-', linewidth=2.5, label='Con variación diaria (A=20)')
    ax5.plot(t, N_sin, 'g--', linewidth=2.5, label='Sin variación (A=0)')
    ax5.axhline(y=N_eq, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax5.set_ylabel('Usuarios en Cola', fontweight='bold')
    ax5.set_title('Efecto de Variación Diaria', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========== Subplot 6: Tabla de análisis ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    data = []
    for xi in [0.1, 0.3, 0.7, 1.0]:
        if xi < 1:
            omega_d = omega_0 * np.sqrt(1 - xi**2)
            T_d = 2 * np.pi / omega_d
            tau = 1 / (xi * omega_0)
            tipo = 'Sub'
        else:
            T_d = 0
            tau = 1 / omega_0
            tipo = 'Crítico'
        data.append([f'{xi}', tipo, f'{tau:.1f}', f'{T_d:.1f}' if T_d > 0 else '-'])
    
    table = ax6.table(cellText=data,
                      colLabels=['ξ', 'Tipo', 'τ (min)', 'T oscil.'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.25, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax6.text(0.5, 0.85, 'Análisis de Amortiguamiento', 
             ha='center', fontsize=12, fontweight='bold', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig('simulacion_sistema_colas.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: simulacion_sistema_colas.png")
    
    # Análisis numérico
    print("\nRESULTADOS NUMÉRICOS:")
    print(f"  Equilibrio teórico: {N_eq:.0f} usuarios")
    for xi in xis:
        if xi < 1:
            tau = 1 / (xi * omega_0)
            print(f"  ξ = {xi:4.1f} → Tiempo de estabilización: ~{5*tau:.0f} min")
    
    return fig

# ============================================================================
# EJECUTAR TODAS LAS SIMULACIONES
# ============================================================================

if __name__ == "__main__":
    # Ejecutar simulación 1
    fig1 = simular_gradiente_descendente()
    
    # Ejecutar simulación 2
    fig2 = simular_sistema_colas()
    
    # Mostrar gráficos
    plt.show()
    
    print("\n" + "="*70)
    print("SIMULACIONES COMPLETADAS EXITOSAMENTE")
    print("="*70)
    print("\nArchivos generados:")
    print("  ✓ simulacion_gradiente_descendente.png")
    print("  ✓ simulacion_sistema_colas.png")
    print("\nPuedes insertar estas imágenes en tu informe.")
