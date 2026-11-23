# Resumen en español del informe (versión 1.0.1, noviembre de 2026)

El documento completo, en inglés para poder publicarlo, está en [preprint.md](preprint.md).

## De qué trata

Un autor publicó en 2026 (D. Wachs, arXiv) que, entre las curvas elípticas de las tablas de
Cremona, los coeficientes aₚ en primos pequeños se relacionan con las piezas de la fórmula de
Birch y Swinnerton-Dyer. Lo único que mantiene tras sus propias correcciones es una correlación
positiva entre aₚ y el período real Ω cuando se fija L(1). Hicimos una **auditoría
independiente**:

- usamos 437 226 clases de isogenia, con una curva por clase;
- condicionamos con barajados dentro de intervalos finos;
- trabajamos con estratos exactos;
- descubrimos en una ventana de conductores y confirmamos en otra distinta;
- registramos las predicciones antes de cada análisis;
- y recalculamos las cifras clave con código independiente.

## Resultados principales

1. **La correlación existe**: +0,049 en p = 3, rango 0. También aparece en rango 1, que nadie
   había estudiado.
2. **En rango 0 se explica por la contabilidad de BSD y un efecto colisionador.** Al fijar L(1),
   la correlación pasa por Tamagawa, y los factores de Euler de los primos malos compiten con los
   de los primos buenos. En las curvas sin torsión, ese factor explica entre el 79 % y el 94 %
   del acoplamiento en rango 0, y entre el 69 % y el 83 % en rango 1.
3. **En rangos 1 y 2 es una paradoja de Simpson**: positiva al mezclar grupos, negativa dentro
   de cada grupo exacto (12 de 12 casos).
4. **La correlación negativa dentro de cada grupo tiene dos causas.**
   - El mismo colisionador, sobre todo en rango 1 con conductores entre 50 000 y 100 000.
   - Un vínculo directo entre aₚ y el exceso de discriminante |Δ|/N, sobre todo en rango 2.
   - Controlando ambas, en la ventana de confirmación los 6 casos quedan por debajo de 0,014;
     cinco son compatibles con cero y uno se pasa un poco (z ≈ 3,5). En la ventana de conductores
     pequeños el control no funciona tan limpio.

**Conclusión:** no hay un vínculo directo entre aₚ y el período; lo que se ve es contabilidad
de BSD, más condicionamiento, más composición.

## Resultados laterales

- Un error en la documentación oficial de los archivos de Cremona (`allgens`), que Cremona ya
  corrigió a partir de nuestra propuesta.
- Un resultado negativo: Sha no cambia los perfiles de aₚ en rango 1.
- En rango 1, Sha ≥ 4 casi nunca aparece en la curva óptima de cada clase, pero sí en sus
  isógenas.
- Tres regularidades sin explicar:
  - exceso de paridad impar en conductores pequeños;
  - una jerarquía entre la 2-torsión módulo p y Tamagawa;
  - L″(1) crece con |Δ| en rango 2.

## Lo que este trabajo NO es

No es un avance hacia la demostración de BSD. Es un trabajo empírico que aclara qué hay detrás
de unas correlaciones publicadas recientemente y corrige su interpretación.
