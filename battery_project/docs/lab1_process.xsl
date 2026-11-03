<?xml version="1.0" encoding="UTF-8"?>
<!-- XSLT 1.0: docs/lab1_process.xml → docs/lab1_process.html (офлайн, без CDN) -->
<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:output method="html" encoding="UTF-8" indent="yes"/>

  <!-- ================= корень ================= -->
  <xsl:template match="/lab1_process">
    <html lang="ru">
      <head>
        <meta charset="UTF-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title><xsl:value-of select="metadata/title"/></title>
        <link rel="stylesheet" href="lab1_process.css"/>
      </head>
      <body>
        <header id="top">
          <h1><xsl:value-of select="metadata/title"/></h1>
          <p class="meta">
            <xsl:value-of select="metadata/author"/> ·
            <xsl:value-of select="metadata/date"/> ·
            версия схемы <xsl:value-of select="metadata/version"/>
          </p>
        </header>

        <!-- ================= оглавление ================= -->
        <nav class="toc">
          <h2>Содержание</h2>
          <ol>
            <li><a href="#s1">Метаданные</a></li>
            <li><a href="#s2">Хронология</a></li>
            <li><a href="#s3">Сущности</a></li>
            <li><a href="#s4">Файлы</a></li>
            <li><a href="#s5">Формулы</a></li>
            <li><a href="#s6">Процессы</a></li>
            <li><a href="#s7">Проверки</a></li>
            <li><a href="#s8">Результаты</a></li>
            <li><a href="#s9">Базовые предсказатели</a></li>
            <li><a href="#s10">Ссылки</a></li>
            <li><a href="#s11">Приложение</a></li>
          </ol>
        </nav>

        <!-- ================= 1. Метаданные ================= -->
        <section id="s1">
          <h2>1. Метаданные</h2>
          <table class="kv">
            <tr><th>Поле</th><th>Значение</th></tr>
            <tr><td>Датасет</td><td><xsl:value-of select="metadata/dataset"/></td></tr>
            <tr><td>Число батарей</td><td><xsl:value-of select="metadata/n_batteries"/></td></tr>
            <tr><td>Разрядных циклов</td><td><xsl:value-of select="metadata/n_discharge_cycles"/></td></tr>
            <tr><td>Тип задачи</td><td><xsl:value-of select="metadata/task_type"/></td></tr>
            <tr><td>Стек</td><td><xsl:value-of select="metadata/stack"/></td></tr>
            <tr><td>seed</td><td><xsl:value-of select="metadata/seed"/></td></tr>
          </table>
        </section>

        <!-- ================= 2. Хронология ================= -->
        <section id="s2">
          <h2>2. Хронология (<xsl:value-of select="count(chronology/stage)"/> этапов)</h2>
          <xsl:for-each select="chronology/stage">
            <div class="stage">
              <h3>
                <span class="stage-num">Этап <xsl:value-of select="@id"/>.</span>
                <xsl:value-of select="@name"/>
              </h3>
              <p class="desc"><xsl:value-of select="description"/></p>

              <xsl:if test="inputs">
                <p class="io"><strong>Входы:</strong>
                  <xsl:for-each select="inputs/input">
                    <span class="io-item"><xsl:value-of select="."/></span>
                  </xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="outputs">
                <p class="io"><strong>Выходы:</strong>
                  <xsl:for-each select="outputs/output">
                    <span class="io-item"><xsl:value-of select="."/></span>
                  </xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="files">
                <p class="io"><strong>Файлы:</strong>
                  <xsl:for-each select="files/file">
                    <code class="io-item"><xsl:value-of select="."/></code>
                  </xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="formulas">
                <p class="io"><strong>Формулы:</strong>
                  <xsl:for-each select="formulas/formula">
                    <code class="io-item"><xsl:value-of select="."/></code>
                  </xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="refs">
                <p class="io"><strong>Ссылки:</strong>
                  <xsl:for-each select="refs/ref">
                    <xsl:choose>
                      <xsl:when test="@href">
                        <a class="io-item" href="{@href}"><xsl:value-of select="."/></a>
                      </xsl:when>
                      <xsl:otherwise>
                        <span class="io-item"><xsl:value-of select="."/></span>
                      </xsl:otherwise>
                    </xsl:choose>
                  </xsl:for-each>
                </p>
              </xsl:if>

              <p class="dur"><strong>Длительность:</strong> <xsl:value-of select="duration"/></p>
            </div>
          </xsl:for-each>
        </section>

        <!-- ================= 3. Сущности ================= -->
        <section id="s3">
          <h2>3. Сущности</h2>

          <h3>3.1 Батареи (RUL-поднабор, 8 шт.)</h3>
          <table>
            <tr><th>ID</th><th>Ambient, °C</th><th>Циклов</th><th>Категория</th></tr>
            <xsl:for-each select="entities/entity[@type='battery']">
              <tr>
                <td><xsl:value-of select="id"/></td>
                <td><xsl:value-of select="ambient_temp"/></td>
                <td><xsl:value-of select="total_cycles"/></td>
                <td><xsl:value-of select="category"/></td>
              </tr>
            </xsl:for-each>
          </table>

          <h3>3.2 Признаки</h3>
          <table>
            <tr><th>Признак</th><th>Формула</th><th>Ед.</th><th>Источник</th></tr>
            <xsl:for-each select="entities/entity[@type='feature']">
              <tr>
                <td><code><xsl:value-of select="name"/></code></td>
                <td><xsl:value-of select="formula"/></td>
                <td><xsl:value-of select="unit"/></td>
                <td><xsl:value-of select="source"/></td>
              </tr>
            </xsl:for-each>
          </table>

          <h3>3.3 Цели</h3>
          <table>
            <tr><th>Цель</th><th>Ед.</th><th>Формула</th></tr>
            <xsl:for-each select="entities/entity[@type='target']">
              <tr>
                <td><code><xsl:value-of select="name"/></code></td>
                <td><xsl:value-of select="unit"/></td>
                <td><xsl:value-of select="formula"/></td>
              </tr>
            </xsl:for-each>
          </table>

          <h3>3.4 Модели (MLP)</h3>
          <table>
            <tr><th>Модель</th><th>Архитектура</th><th>Параметры</th><th>Loss</th><th>Метрика</th></tr>
            <xsl:for-each select="entities/entity[@type='model']">
              <tr>
                <td><code><xsl:value-of select="name"/></code></td>
                <td><xsl:value-of select="architecture"/></td>
                <td><xsl:value-of select="params"/></td>
                <td><xsl:value-of select="loss"/></td>
                <td><xsl:value-of select="metric"/></td>
              </tr>
            </xsl:for-each>
          </table>
        </section>

        <!-- ================= 4. Файлы ================= -->
        <section id="s4">
          <h2>4. Файлы</h2>
          <table>
            <tr><th>ID</th><th>Путь</th><th>Тип</th><th>Размер (строк × колонок)</th><th>Содержимое</th><th>Создатель</th><th>Используется</th></tr>
            <xsl:for-each select="files/file">
              <tr>
                <td><code><xsl:value-of select="@id"/></code></td>
                <td><code><xsl:value-of select="path"/></code></td>
                <td><xsl:value-of select="type"/></td>
                <td>
                  <xsl:choose>
                    <xsl:when test="rows">
                      <xsl:value-of select="rows"/> × <xsl:value-of select="cols"/>
                    </xsl:when>
                    <xsl:otherwise>—</xsl:otherwise>
                  </xsl:choose>
                </td>
                <td><xsl:value-of select="content"/></td>
                <td><code><xsl:value-of select="created_by"/></code></td>
                <td>
                  <xsl:for-each select="used_by">
                    <code class="io-item"><xsl:value-of select="."/></code>
                  </xsl:for-each>
                </td>
              </tr>
            </xsl:for-each>
          </table>
        </section>

        <!-- ================= 5. Формулы ================= -->
        <section id="s5">
          <h2>5. Формулы</h2>
          <table>
            <tr><th>ID</th><th>Название</th><th>LaTeX</th><th>Plain</th><th>Используется</th></tr>
            <xsl:for-each select="formulas/formula">
              <tr>
                <td><code><xsl:value-of select="@id"/></code></td>
                <td><xsl:value-of select="name"/></td>
                <td><code class="formula">\(<xsl:value-of select="latex"/>\)</code></td>
                <td><code class="formula"><xsl:value-of select="plain"/></code></td>
                <td><xsl:value-of select="used_in"/></td>
              </tr>
            </xsl:for-each>
          </table>
        </section>

        <!-- ================= 6. Процессы ================= -->
        <section id="s6">
          <h2>6. Процессы</h2>
          <xsl:for-each select="processes/process">
            <div class="stage">
              <h3><code><xsl:value-of select="@id"/></code> — <xsl:value-of select="name"/></h3>
              <p class="desc"><xsl:value-of select="description"/></p>

              <xsl:if test="inputs">
                <p class="io"><strong>Входы:</strong>
                  <xsl:for-each select="inputs/input"><span class="io-item"><xsl:value-of select="."/></span></xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="outputs">
                <p class="io"><strong>Выходы:</strong>
                  <xsl:for-each select="outputs/output"><span class="io-item"><xsl:value-of select="."/></span></xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="steps">
                <ol class="steps">
                  <xsl:for-each select="steps/step"><li><xsl:value-of select="."/></li></xsl:for-each>
                </ol>
              </xsl:if>

              <xsl:if test="rules">
                <ul class="rules">
                  <xsl:for-each select="rules/rule"><li><xsl:value-of select="."/></li></xsl:for-each>
                </ul>
              </xsl:if>

              <xsl:if test="features">
                <p class="io"><strong>Признаки:</strong>
                  <xsl:for-each select="features/feature"><code class="io-item"><xsl:value-of select="."/></code></xsl:for-each>
                </p>
              </xsl:if>

              <xsl:if test="hyperparams">
                <table class="hp">
                  <tr><th>Гиперпараметр</th><th>Значение</th></tr>
                  <xsl:for-each select="hyperparams/param">
                    <tr><td><code><xsl:value-of select="@name"/></code></td><td><xsl:value-of select="."/></td></tr>
                  </xsl:for-each>
                </table>
              </xsl:if>

              <xsl:if test="refs">
                <p class="io"><strong>Исходники:</strong>
                  <xsl:for-each select="refs/ref"><code class="io-item"><xsl:value-of select="."/></code></xsl:for-each>
                </p>
              </xsl:if>
            </div>
          </xsl:for-each>
        </section>

        <!-- ================= 7. Проверки ================= -->
        <section id="s7">
          <h2>7. Проверки</h2>
          <table>
            <tr><th>ID</th><th>Название</th><th>Описание</th><th>Скрипт</th><th>Метрика</th><th>Статус</th></tr>
            <xsl:for-each select="checks/check">
              <tr>
                <td><code><xsl:value-of select="@id"/></code></td>
                <td><xsl:value-of select="name"/></td>
                <td><xsl:value-of select="description"/></td>
                <td><code><xsl:value-of select="script"/></code></td>
                <td><xsl:value-of select="metric"/></td>
                <td>
                  <xsl:choose>
                    <xsl:when test="status = 'PASSED'"><span class="badge ok"><xsl:value-of select="status"/></span></xsl:when>
                    <xsl:otherwise><span class="badge bad"><xsl:value-of select="status"/></span></xsl:otherwise>
                  </xsl:choose>
                </td>
              </tr>
            </xsl:for-each>
          </table>
        </section>

        <!-- ================= 8. Результаты ================= -->
        <section id="s8">
          <h2>8. Результаты</h2>
          <xsl:for-each select="results/result">
            <h3>Задача <code><xsl:value-of select="@task"/></code> — цель <xsl:value-of select="target"/></h3>
            <table class="res">
              <tr>
                <th>Сплит</th>
                <xsl:for-each select="metrics/metric"><th><xsl:value-of select="@name"/></th></xsl:for-each>
              </tr>
              <tr>
                <td>Батарей / train / val / test:
                  <xsl:value-of select="n_batteries"/> /
                  <xsl:value-of select="n_train"/> /
                  <xsl:value-of select="n_val"/> /
                  <xsl:value-of select="n_test"/>
                </td>
                <xsl:for-each select="metrics/metric"><td><xsl:value-of select="."/></td></xsl:for-each>
              </tr>
            </table>
            <xsl:if test="per_battery">
              <p class="io"><strong>Per-battery MAE (test):</strong>
                <xsl:for-each select="per_battery/battery">
                  <span class="io-item"><xsl:value-of select="@id"/>: <xsl:value-of select="."/></span>
                </xsl:for-each>
              </p>
            </xsl:if>
          </xsl:for-each>
        </section>

        <!-- ================= 9. Базовые предсказатели ================= -->
        <section id="s9">
          <h2>9. Базовые предсказатели (baseline)</h2>
          <p class="desc">Один и тот же сплит по батареям и один и тот же препроцессинг для всех
            четырёх методов. У константы R² = 0 по построению; отрицательное R² — хуже константы.</p>
          <xsl:for-each select="baselines/baseline">
            <div class="stage version">
              <h3><xsl:value-of select="@name"/></h3>
              <p class="desc"><xsl:value-of select="description"/></p>

              <xsl:if test="metrics/task">
                <table>
                  <tr><th>Задача</th><th>R²</th><th>MAE</th><th>RMSE</th></tr>
                  <xsl:for-each select="metrics/task">
                    <tr>
                      <td><xsl:value-of select="@id"/></td>
                      <td><xsl:value-of select="@r2"/></td>
                      <td><xsl:value-of select="@mae"/></td>
                      <td><xsl:value-of select="@rmse"/></td>
                    </tr>
                  </xsl:for-each>
                </table>
              </xsl:if>

              <p class="note"><xsl:value-of select="note"/></p>
            </div>
          </xsl:for-each>
        </section>

        <!-- ================= 10. Ссылки ================= -->
        <section id="s10">
          <h2>10. Ссылки</h2>
          <ul class="refs">
            <xsl:for-each select="references/ref">
              <li>
                <xsl:value-of select="title"/>
                <xsl:choose>
                  <xsl:when test="starts-with(url, 'file://')">
                    — <a class="local" href="{substring-after(url, 'file://')}"><code><xsl:value-of select="substring-after(url, 'file://')"/></code></a>
                  </xsl:when>
                  <xsl:otherwise>
                    — <a href="{url}" rel="noopener"><code><xsl:value-of select="url"/></code></a>
                  </xsl:otherwise>
                </xsl:choose>
              </li>
            </xsl:for-each>
          </ul>
        </section>

        <!-- ================= 11. Приложение ================= -->
        <section id="s11">
          <h2>11. Приложение</h2>

          <h3>11.1 Дерево файлов</h3>
          <pre class="tree"><xsl:value-of select="appendix/files_tree/pre"/></pre>

          <h3>11.2 Таблица метрик (сводная)</h3>
          <table>
            <xsl:for-each select="appendix/metrics_table/row">
              <xsl:choose>
                <xsl:when test="position() = 1">
                  <tr class="thead">
                    <xsl:for-each select="cell"><th><xsl:value-of select="."/></th></xsl:for-each>
                  </tr>
                </xsl:when>
                <xsl:otherwise>
                  <tr>
                    <xsl:for-each select="cell"><td><xsl:value-of select="."/></td></xsl:for-each>
                  </tr>
                </xsl:otherwise>
              </xsl:choose>
            </xsl:for-each>
          </table>

          <h3>11.3 Графики</h3>
          <table>
            <tr><th>ID</th><th>Название</th><th>Файл</th></tr>
            <xsl:for-each select="appendix/graphs_list/graph">
              <tr>
                <td><xsl:value-of select="id"/></td>
                <td><xsl:value-of select="title"/></td>
                <td><code><xsl:value-of select="file"/></code></td>
              </tr>
            </xsl:for-each>
          </table>

          <h3>11.4 Тесты</h3>
          <table>
            <tr><th>Набор</th><th>Pytest</th><th>Результат</th></tr>
            <xsl:for-each select="appendix/tests_list/test">
              <tr>
                <td><xsl:value-of select="name"/></td>
                <td><code><xsl:value-of select="pytest_id"/></code></td>
                <td>
                  <xsl:choose>
                    <xsl:when test="contains(result, 'passed') or result = 'PASSED'"><span class="badge ok"><xsl:value-of select="result"/></span></xsl:when>
                    <xsl:otherwise><span class="badge bad"><xsl:value-of select="result"/></span></xsl:otherwise>
                  </xsl:choose>
                </td>
              </tr>
            </xsl:for-each>
          </table>

          <h3>11.5 Команды</h3>
          <table>
            <tr><th>Задача</th><th>Команда</th><th>Описание</th></tr>
            <xsl:for-each select="appendix/commands/command">
              <tr>
                <td><xsl:value-of select="name"/></td>
                <td><code><xsl:value-of select="cmd"/></code></td>
                <td><xsl:value-of select="description"/></td>
              </tr>
            </xsl:for-each>
          </table>
        </section>

        <footer>
          <p>Сгенерировано из <code>docs/lab1_process.xml</code> через <code>scripts/xml_to_html.py</code> (lxml XSLT). Все ресурсы локальные.</p>
          <p><a href="#top">↑ Наверх</a></p>
        </footer>
      </body>
    </html>
  </xsl:template>

</xsl:stylesheet>