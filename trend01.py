import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yfin
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import bt

# BacktestAnalyzer 클래스 정의 (수정된 버전)
class BacktestAnalyzer:
    def __init__(self):
        # 기본 설정값
        self.symbol = "BTC-USD"
        self.years = 5
        self.initial_capital = 100000000
        self.use_absolute_momentum = True
        self.sma_periods = [120]
        self.ema_short = 7
        self.ema_long = 30
        self.triple_short = 7
        self.triple_mid = 30
        self.triple_long = 60
        self.weights = {
            'macd': 0.3,
            'ema': 0.3,
            'triple': 0.3
        }
        self.strategy_enabled = {
            'macd': True,
            'ema': True,
            'triple': True
        }

    def set_parameters(self, **kwargs):
        """설정값 변경"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # 활성화된 전략이 하나도 없는지 확인
        if isinstance(self.strategy_enabled, dict):
            if not any(self.strategy_enabled.values()):
                st.error("최소한 하나의 전략을 활성화해야 합니다.")
                return False
        
        # 전략 가중치 정규화
        if isinstance(self.weights, dict) and isinstance(self.strategy_enabled, dict):
            active_weights = {k: v for k, v in self.weights.items() if self.strategy_enabled.get(k, False)}
            total_weight = sum(active_weights.values())
            
            if total_weight > 0:  # 0으로 나누는 오류 방지
                for k in active_weights:
                    self.weights[k] = active_weights[k] / total_weight
        
        return True

    def run_backtest(self):
        """백테스트 실행"""
        # 활성화된 전략이 없으면 중단
        active_strategies = [k for k, v in self.strategy_enabled.items() if v]
        if not active_strategies:
            st.error("최소한 하나의 전략을 활성화해야 합니다.")
            return None
            
        # 날짜 설정
        today = datetime.today().strftime("%Y-%m-%d")
        fr = datetime.now() - timedelta(self.years*365)
        fr = fr.strftime("%Y-%m-%d")

        # 데이터 다운로드
        try:
            st.info(f"데이터 다운로드 중... ({self.symbol})")
            df = yfin.download(self.symbol, start=fr, end=today)
            if df.empty:
                st.error("유효한 데이터가 없습니다.")
                return None
            
            # 성공 메시지
            st.success(f"{self.symbol} 데이터 다운로드 완료")
        except Exception as e:
            st.error(f"데이터 다운로드 중 오류 발생: {str(e)}")
            return None

        # 기본 데이터 준비
        df = df[['Close']].copy()
        df = df.ffill().bfill()

        # 절대 모멘텀 시그널 계산
        if self.use_absolute_momentum and self.sma_periods:
            sma_signals = pd.DataFrame(index=df.index)
            for period in self.sma_periods:
                sma = df['Close'].rolling(window=period).mean()
                sma = sma.ffill().bfill()
                sma_signals[f'sma_{period}'] = df['Close'] > sma
            absolute_momentum = sma_signals.all(axis=1)
        else:
            absolute_momentum = pd.Series(True, index=df.index)

        # 활성화된 전략 목록 생성
        active_strategies = []
        strategy_signals = {}
        stockdata = pd.DataFrame(index=df.index)  # 초기화

        # MACD 전략
        df_macd = None
        macd = None
        macd_signal = None
        
        if self.strategy_enabled.get('macd', False):
            df_macd = df.copy()
            df_macd.columns = ['macd']
            st_line = ta.ema(df_macd['macd'], 9)
            mt_line = ta.ema(df_macd['macd'], 26)
            macd = st_line - mt_line
            macd_signal = ta.ema(macd, 9)
            sign = macd - macd_signal
            
            # 시그널 생성
            signal_macd = pd.DataFrame(index=df.index)
            signal_macd['macd'] = self.weights.get('macd', 0.3) * (sign > 0) * absolute_momentum
            signal_macd = signal_macd.fillna(0)
            
            active_strategies.append('macd')
            strategy_signals['macd'] = signal_macd['macd']
            stockdata['macd'] = df_macd['macd']
        else:
            signal_macd = pd.DataFrame(index=df.index, columns=['macd'])
            signal_macd['macd'] = 0

        # EMA 크로스 전략
        ema_st = None
        ema_lt = None
        
        if self.strategy_enabled.get('ema', False):
            df_ema = df.copy()
            df_ema.columns = ['ema']
            ema_st = ta.ema(df_ema['ema'], self.ema_short)
            ema_lt = ta.ema(df_ema['ema'], self.ema_long)
            sign = ema_st - ema_lt
            
            # 시그널 생성
            signal_ema = pd.DataFrame(index=df.index)
            signal_ema['ema'] = self.weights.get('ema', 0.3) * (sign > 0) * absolute_momentum
            signal_ema = signal_ema.fillna(0)
            
            active_strategies.append('ema')
            strategy_signals['ema'] = signal_ema['ema']
            stockdata['ema'] = df_ema['ema']
        else:
            signal_ema = pd.DataFrame(index=df.index, columns=['ema'])
            signal_ema['ema'] = 0

        # 삼중 EMA 전략
        ema_short_t = None
        ema_mid_t = None
        ema_long_t = None
        
        if self.strategy_enabled.get('triple', False):
            df_triple = df.copy()
            df_triple.columns = ['triple']
            ema_short_t = ta.ema(df_triple['triple'], self.triple_short).fillna(0)
            ema_mid_t = ta.ema(df_triple['triple'], self.triple_mid).fillna(0)
            ema_long_t = ta.ema(df_triple['triple'], self.triple_long).fillna(0)

            condition1 = (ema_short_t > ema_mid_t) & (ema_mid_t > ema_long_t)
            condition2 = (ema_long_t > ema_mid_t) & (ema_mid_t > ema_short_t)
            condition3 = ~(condition1 | condition2)

            # 시그널 생성
            signal_triple = pd.DataFrame(index=df.index)
            signal_triple['triple'] = np.select(
                [condition1, condition2, condition3],
                [self.weights.get('triple', 0.3), self.weights.get('triple', 0.3)*0.33, self.weights.get('triple', 0.3)*0.67],
                default=0
            ) * absolute_momentum
            
            active_strategies.append('triple')
            strategy_signals['triple'] = signal_triple['triple']
            stockdata['triple'] = df_triple['triple']
        else:
            signal_triple = pd.DataFrame(index=df.index, columns=['triple'])
            signal_triple['triple'] = 0

        # 모든 전략이 비활성화된 경우 처리
        if not active_strategies:
            st.warning("최소한 하나의 전략을 활성화해야 합니다.")
            return None

        # 데이터 병합 (이미 활성화된 전략만 stockdata에 추가되어 있음)
        b_signal = pd.concat([signal_macd, signal_ema, signal_triple], axis=1)

        # 벤치마크용 Buy & Hold 데이터 추가
        stockdata['buy_hold'] = df['Close']  # Buy & Hold 데이터 추가
        b_signal['buy_hold'] = 1.0  # Buy & Hold는 항상 100% 투자

        # 전략 설정
        # 1. 멀티 전략
        try:
            multi_strategy = bt.Strategy('Multi_Strategy', [
                bt.algos.WeighTarget(b_signal[active_strategies]),
                bt.algos.Rebalance()
            ])

            # 2. Buy & Hold 전략
            buy_hold_strategy = bt.Strategy('Buy_Hold', [
                bt.algos.WeighTarget(b_signal[['buy_hold']]),
                bt.algos.Rebalance()
            ])

            # 백테스트 설정
            multi_backtest = bt.Backtest(multi_strategy, stockdata[active_strategies])
            buy_hold_backtest = bt.Backtest(buy_hold_strategy, stockdata[['buy_hold']])

            # 백테스트 실행
            self.result = bt.run(multi_backtest, buy_hold_backtest)
            self.df = df
            self.b_signal = b_signal[active_strategies]  # Buy & Hold 제외
            self.indicators = {
                'macd_line': macd,
                'macd_signal': macd_signal,
                'ema_st': ema_st,
                'ema_lt': ema_lt,
                'ema_short_t': ema_short_t,
                'ema_mid_t': ema_mid_t,
                'ema_long_t': ema_long_t
            }
            self.strategy_signals = strategy_signals
            self.active_strategies = active_strategies

            return self.result
            
        except Exception as e:
            st.error(f"백테스트 실행 중 오류 발생: {str(e)}")
            return None

    def get_summary_data(self):
        """투자 성과 요약 데이터 반환"""
        if not hasattr(self, 'result'):
            return None

        try:
            # 멀티 전략 성과
            multi_stats = self.result['Multi_Strategy'].stats
            total_return = float(multi_stats.loc['total_return'])
            cagr = float(multi_stats.loc['cagr'])
            sharpe = float(multi_stats.loc['daily_sharpe'])
            max_dd = float(multi_stats.loc['max_drawdown'])

            # Buy & Hold 성과
            bh_stats = self.result['Buy_Hold'].stats
            bh_total_return = float(bh_stats.loc['total_return'])
            bh_cagr = float(bh_stats.loc['cagr'])
            bh_sharpe = float(bh_stats.loc['daily_sharpe'])
            bh_max_dd = float(bh_stats.loc['max_drawdown'])

            # 성과 비교
            return_diff = total_return - bh_total_return

            summary = {
                "multi_total_return": total_return*100,
                "multi_cagr": cagr*100,
                "multi_sharpe": sharpe,
                "multi_max_dd": max_dd*100,
                "bh_total_return": bh_total_return*100,
                "bh_cagr": bh_cagr*100,
                "bh_sharpe": bh_sharpe,
                "bh_max_dd": bh_max_dd*100,
                "return_diff": return_diff*100,
                "max_dd_diff": (bh_max_dd - max_dd)*100,
                "sharpe_diff": sharpe - bh_sharpe,
                "return_diff_status": "우수" if return_diff > 0 else "열등",
                "max_dd_diff_status": "우수" if (bh_max_dd - max_dd) > 0 else "열등",
                "sharpe_diff_status": "우수" if (sharpe - bh_sharpe) > 0 else "열등"
            }
            
            return summary
            
        except Exception as e:
            st.error(f"성과 데이터 계산 중 오류 발생: {str(e)}")
            return None

    def get_monthly_returns_data(self):
        """월별 수익률 분석 데이터 반환"""
        if not hasattr(self, 'result'):
            return None

        try:
            # 멀티 전략 월별 수익률 계산
            multi_returns = self.result['Multi_Strategy'].prices.pct_change()
            multi_monthly_returns = multi_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if isinstance(multi_monthly_returns, pd.DataFrame) and multi_monthly_returns.shape[1] > 0:
                multi_monthly_returns = multi_monthly_returns.iloc[:,0]

            # Buy & Hold 월별 수익률 계산
            bh_returns = self.result['Buy_Hold'].prices.pct_change()
            bh_monthly_returns = bh_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if isinstance(bh_monthly_returns, pd.DataFrame) and bh_monthly_returns.shape[1] > 0:
                bh_monthly_returns = bh_monthly_returns.iloc[:,0]

            # 월별 거래 포지션 계산
            total_position = self.b_signal.sum(axis=1)
            position_monthly = total_position.resample('M').mean()

            # 실제 거래가 발생한 달만 필터링 (포지션이 1% 이상인 경우)
            monthly_trades = pd.DataFrame({
                'multi_returns': multi_monthly_returns,
                'bh_returns': bh_monthly_returns,
                'position': position_monthly
            })
            traded_months = monthly_trades[monthly_trades['position'] >= 0.01]

            if len(traded_months) > 0:
                # 거래가 발생한 달의 통계 계산
                winning_months = (traded_months['multi_returns'] > 0).mean()
                avg_win = traded_months[traded_months['multi_returns'] > 0]['multi_returns'].mean()
                avg_loss = traded_months[traded_months['multi_returns'] < 0]['multi_returns'].mean()
                
                # avg_loss가 NaN인 경우 (모든 달이 수익인 경우) 처리
                if pd.isna(avg_loss):
                    avg_loss = 0

                # Buy & Hold 통계
                bh_winning_months = (traded_months['bh_returns'] > 0).mean()

                # 멀티 전략 vs Buy & Hold 비교
                better_than_bh = (traded_months['multi_returns'] > traded_months['bh_returns']).mean()

                # 월간 예상 수익 계산
                trading_months_return = float(np.power(1 + avg_win, winning_months) *
                                           np.power(1 + avg_loss, (1-winning_months)) - 1)
                expected_monthly_return = int(trading_months_return * self.initial_capital)

                # 연간 실제 거래 개월 수 계산
                total_months = (traded_months.index[-1].year - traded_months.index[0].year) * 12 + \
                              traded_months.index[-1].month - traded_months.index[0].month + 1
                trading_ratio = len(traded_months) / total_months
                months_per_year = min(12 * trading_ratio, 12)

                monthly_data = {
                    "expected_monthly_return_pct": trading_months_return*100,
                    "expected_monthly_return_krw": expected_monthly_return,
                    "winning_months_pct": winning_months*100,
                    "bh_winning_months_pct": bh_winning_months*100,
                    "better_than_bh_pct": better_than_bh*100,
                    "months_per_year": months_per_year,
                    "total_months": total_months,
                    "traded_months": len(traded_months),
                    "trading_ratio_pct": trading_ratio*100,
                    "monthly_returns": traded_months
                }
                
                return monthly_data
            else:
                return {"error": "분석 기간 동안 거래가 발생하지 않았습니다."}
                
        except Exception as e:
            st.error(f"월별 수익률 계산 중 오류 발생: {str(e)}")
            return {"error": f"월별 수익률 계산 중 오류 발생: {str(e)}"}

    def create_cumulative_returns_chart(self):
        """누적 수익률 비교 차트 생성 (Plotly)"""
        if not hasattr(self, 'result'):
            return None

        try:
            # 멀티 전략 누적 수익률
            multi_returns = self.result['Multi_Strategy'].prices / self.result['Multi_Strategy'].prices.iloc[0] * 100 - 100
            
            # Buy & Hold 누적 수익률
            bh_returns = self.result['Buy_Hold'].prices / self.result['Buy_Hold'].prices.iloc[0] * 100 - 100
            
            # Plotly 차트 생성
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=multi_returns.index, 
                y=multi_returns, 
                mode='lines',
                name='추세추종 전략',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=bh_returns.index, 
                y=bh_returns, 
                mode='lines',
                name='Buy & Hold',
                line=dict(width=2, dash='dash')
            ))
            
            fig.add_shape(
                type="line",
                x0=multi_returns.index[0],
                y0=0,
                x1=multi_returns.index[-1],
                y1=0,
                line=dict(color="red", width=1, dash="dash"),
            )
            
            fig.update_layout(
                title="누적 수익률 비교 (%)",
                xaxis_title="날짜",
                yaxis_title="수익률 (%)",
                legend_title="전략",
                template="plotly_white",
                hovermode="x unified"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"누적 수익률 차트 생성 중 오류 발생: {str(e)}")
            return None

    def create_detailed_analysis_chart(self):
        """종합 분석 차트 생성 (Matplotlib 사용)"""
        if not hasattr(self, 'result') or not hasattr(self, 'df'):
            return None
            
        try:
            # 그림 생성
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.1)

            # 가격 차트와 포지션
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(self.df.index, self.df['Close'], label='Price', color='black', alpha=0.7)

            # SMA 라인 추가
            if self.use_absolute_momentum and self.sma_periods:
                for period in self.sma_periods:
                    sma = self.df['Close'].rolling(window=period).mean()
                    sma = sma.ffill().bfill()
                    ax1.plot(self.df.index, sma, '--', label=f'SMA {period}', alpha=0.5)

            ax1.set_title(f'{self.symbol} Price and Positions')
            ax1.grid(True)
            ax1.legend(loc='upper left')

            # 총 포지션 표시
            total_position = self.b_signal.sum(axis=1) * 100
            ax1_twin = ax1.twinx()
            ax1_twin.plot(self.df.index, total_position, '--', color='red', label='Total Position %', alpha=0.5)
            ax1_twin.fill_between(self.df.index, 0, total_position, alpha=0.1, color='red')
            ax1_twin.set_ylim(0, 100)
            ax1_twin.set_ylabel('Position %')
            ax1_twin.legend(loc='upper right')

            # 전략 성과 비교
            ax5 = fig.add_subplot(gs[1], sharex=ax1)
            relative_performance = (self.result['Multi_Strategy'].prices / self.result['Multi_Strategy'].prices.iloc[0]) / \
                                (self.result['Buy_Hold'].prices / self.result['Buy_Hold'].prices.iloc[0]) - 1
            ax5.plot(relative_performance.index, relative_performance * 100, label='Multi vs Buy & Hold', color='purple')
            ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax5.fill_between(relative_performance.index, 0, relative_performance * 100,
                            alpha=0.2, color='green', where=(relative_performance > 0))
            ax5.fill_between(relative_performance.index, 0, relative_performance * 100,
                            alpha=0.2, color='red', where=(relative_performance < 0))
            ax5.set_title('Multi Strategy vs Buy & Hold (% Relative Performance)')
            ax5.set_ylabel('%')
            ax5.grid(True)
            ax5.legend()

            current_row = 2  # 현재 사용 중인 행 추적

            # 기술적 지표들 - MACD
            if 'macd' in self.active_strategies and self.indicators['macd_line'] is not None:
                ax2 = fig.add_subplot(gs[current_row], sharex=ax1)
                ax2.plot(self.df.index, self.indicators['macd_line'], label='MACD', color='blue')
                ax2.plot(self.df.index, self.indicators['macd_signal'], label='Signal', color='red')
                ax2.fill_between(self.df.index,
                                self.indicators['macd_line'] - self.indicators['macd_signal'],
                                color='gray', alpha=0.3)
                ax2.set_title('MACD')
                ax2.grid(True)
                ax2.legend()
                current_row += 1

            # EMA 크로스 지표
            if 'ema' in self.active_strategies and self.indicators['ema_st'] is not None:
                ax3 = fig.add_subplot(gs[current_row], sharex=ax1)
                ax3.plot(self.df.index, self.indicators['ema_st'],
                        label=f'EMA {self.ema_short}', color='blue')
                ax3.plot(self.df.index, self.indicators['ema_lt'],
                        label=f'EMA {self.ema_long}', color='red')
                ax3.set_title('EMA Cross')
                ax3.grid(True)
                ax3.legend()
                current_row += 1

            # 삼중 EMA 지표
            if 'triple' in self.active_strategies and self.indicators['ema_short_t'] is not None:
                ax4 = fig.add_subplot(gs[current_row], sharex=ax1)
                ax4.plot(self.df.index, self.indicators['ema_short_t'],
                        label=f'EMA {self.triple_short}', color='blue')
                ax4.plot(self.df.index, self.indicators['ema_mid_t'],
                        label=f'EMA {self.triple_mid}', color='red')
                ax4.plot(self.df.index, self.indicators['ema_long_t'],
                        label=f'EMA {self.triple_long}', color='green')
                ax4.set_title('Triple EMA')
                ax4.grid(True)
                ax4.legend()

            # x축 레이블 숨기기 (마지막 그래프 제외)
            if 'macd' in self.active_strategies:
                plt.setp(ax2.get_xticklabels(), visible=False)
            if 'ema' in self.active_strategies:
                plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp([ax1.get_xticklabels(), ax5.get_xticklabels()], visible=False)

            # 그림 저장 및 스트림릿에 표시
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"분석 차트 생성 중 오류 발생: {str(e)}")
            return None

    def create_monthly_heatmap(self):
        """월별 수익률 히트맵 생성 (Plotly)"""
        if not hasattr(self, 'result'):
            return None
            
        try:
            # 멀티 전략 월별 수익률 계산
            multi_returns = self.result['Multi_Strategy'].prices.pct_change()
            multi_monthly_returns = multi_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if isinstance(multi_monthly_returns, pd.DataFrame) and multi_monthly_returns.shape[1] > 0:
                multi_monthly_returns = multi_monthly_returns.iloc[:,0]
                
            # Buy & Hold 월별 수익률 계산
            bh_returns = self.result['Buy_Hold'].prices.pct_change()
            bh_monthly_returns = bh_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if isinstance(bh_monthly_returns, pd.DataFrame) and bh_monthly_returns.shape[1] > 0:
                bh_monthly_returns = bh_monthly_returns.iloc[:,0]
                
            # 월별 데이터 필터링 (적어도 6개월 이상의 데이터가 있어야 함)
            if len(multi_monthly_returns) < 6 or len(bh_monthly_returns) < 6:
                st.warning("히트맵을 생성하기 위한 충분한 월별 데이터가 없습니다 (최소 6개월 필요).")
                return None
                
            # 멀티 전략 월별 수익률 매트릭스 구성
            multi_returns_df = pd.DataFrame(multi_monthly_returns)
            multi_returns_df.columns = [0]  # 컬럼명 명시적 설정
            multi_returns_df['year'] = multi_returns_df.index.year
            multi_returns_df['month'] = multi_returns_df.index.month
            
            try:
                multi_returns_matrix = multi_returns_df.pivot_table(index='year', columns='month', values=0)
                
                # Buy & Hold 월별 수익률 매트릭스 구성
                bh_returns_df = pd.DataFrame(bh_monthly_returns)
                bh_returns_df.columns = [0]  # 컬럼명 명시적 설정
                bh_returns_df['year'] = bh_returns_df.index.year
                bh_returns_df['month'] = bh_returns_df.index.month
                bh_returns_matrix = bh_returns_df.pivot_table(index='year', columns='month', values=0)
                
                # 서브플롯 생성
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    subplot_titles=("추세추종 전략 월별 수익률", "Buy & Hold 월별 수익률"),
                    vertical_spacing=0.15
                )
                
                # 멀티 전략 히트맵
                z_multi = multi_returns_matrix.values
                fig.add_trace(
                    go.Heatmap(
                        z=z_multi,
                        x=multi_returns_matrix.columns,
                        y=multi_returns_matrix.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=np.round(z_multi * 100, 1),
                        texttemplate="%{text}%",
                        colorbar=dict(title="수익률 %", y=0.75, len=0.45)
                    ),
                    row=1, col=1
                )
                
                # Buy & Hold 히트맵
                z_bh = bh_returns_matrix.values
                fig.add_trace(
                    go.Heatmap(
                        z=z_bh,
                        x=bh_returns_matrix.columns,
                        y=bh_returns_matrix.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=np.round(z_bh * 100, 1),
                        texttemplate="%{text}%",
                        colorbar=dict(title="수익률 %", y=0.25, len=0.45)
                    ),
                    row=2, col=1
                )
                
                # x축 월 이름 설정
                month_names = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
                fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
                
                # 레이아웃 업데이트
                fig.update_layout(
                    height=800,
                    title_text=f"{self.symbol} 월별 수익률 히트맵",
                    template="plotly_white"
                )
                
                return fig
            except Exception as e:
                st.warning(f"히트맵 생성 오류: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {str(e)}")
            return None


# Streamlit 앱 UI 구성
st.set_page_config(page_title="단일종목 추세추종 백테스트", layout="wide")
st.title("📈 단일종목 추세추종 백테스트")
st.markdown("이 앱은 다양한 기술적 지표를 기반으로 한 트레이딩 전략의 백테스트 결과를 제공합니다.")

# 사이드바 설정
st.sidebar.header("📊 백테스트 파라미터")

# 백테스트 분석기 인스턴스 생성
analyzer = BacktestAnalyzer()

# 사이드바 입력 파라미터
symbol = st.sidebar.text_input("티커 심볼", value="BTC-USD")
years = st.sidebar.slider("분석 기간 (년)", min_value=1, max_value=20, value=5)
initial_capital = st.sidebar.number_input("초기 자본금", min_value=1000000, value=100000000, step=1000000)

# 절대 모멘텀 설정
use_absolute_momentum = st.sidebar.checkbox("절대 모멘텀 사용", value=True)
sma_periods = []
if use_absolute_momentum:
    sma_period = st.sidebar.slider("SMA 기간", min_value=10, max_value=200, value=120)
    sma_periods = [sma_period]

# 전략 선택 및 가중치 설정
st.sidebar.header("📈 트레이딩 전략")

# MACD 전략
macd_enabled = st.sidebar.checkbox("MACD 전략 사용", value=True)
macd_weight = 0.3
if macd_enabled:
    macd_weight = st.sidebar.slider("MACD 가중치", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

# EMA 크로스 전략
ema_enabled = st.sidebar.checkbox("EMA 크로스 전략 사용", value=True)
ema_short, ema_long = 7, 30
ema_weight = 0.3
if ema_enabled:
    ema_short = st.sidebar.slider("EMA 단기 기간", min_value=3, max_value=30, value=7)
    ema_long = st.sidebar.slider("EMA 장기 기간", min_value=20, max_value=100, value=30)
    ema_weight = st.sidebar.slider("EMA 가중치", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

# 삼중 EMA 전략
triple_enabled = st.sidebar.checkbox("삼중 EMA 전략 사용", value=True)
triple_short, triple_mid, triple_long = 7, 30, 60
triple_weight = 0.3
if triple_enabled:
    triple_short = st.sidebar.slider("삼중 EMA 단기 기간", min_value=3, max_value=20, value=7)
    triple_mid = st.sidebar.slider("삼중 EMA 중기 기간", min_value=20, max_value=50, value=30)
    triple_long = st.sidebar.slider("삼중 EMA 장기 기간", min_value=50, max_value=150, value=60)
    triple_weight = st.sidebar.slider("삼중 EMA 가중치", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

# 적어도 하나의 전략이 활성화되어 있는지 확인
if not any([macd_enabled, ema_enabled, triple_enabled]):
    st.sidebar.error("최소한 하나의 전략을 활성화해야 합니다!")
    st.error("백테스트를 실행하려면 최소한 하나의 전략을 활성화하세요.")
else:
    # 가중치 정규화
    total_weight = 0
    if macd_enabled: total_weight += macd_weight
    if ema_enabled: total_weight += ema_weight 
    if triple_enabled: total_weight += triple_weight
    
    if total_weight > 0:  # 0으로 나누는 오류 방지
        normalized_weights = {
            'macd': macd_weight / total_weight if macd_enabled else 0,
            'ema': ema_weight / total_weight if ema_enabled else 0,
            'triple': triple_weight / total_weight if triple_enabled else 0
        }
        
        # 정규화된 가중치 보여주기
        st.sidebar.subheader("정규화된 가중치")
        for strategy, weight in normalized_weights.items():
            if weight > 0:
                st.sidebar.text(f"{strategy.upper()}: {weight:.2f}")
    
    # 파라미터 설정
    analyzer.set_parameters(
        symbol=symbol,
        years=years,
        initial_capital=initial_capital,
        use_absolute_momentum=use_absolute_momentum,
        sma_periods=sma_periods,
        ema_short=ema_short,
        ema_long=ema_long,
        triple_short=triple_short,
        triple_mid=triple_mid,
        triple_long=triple_long,
        weights=normalized_weights,
        strategy_enabled={
            'macd': macd_enabled,
            'ema': ema_enabled,
            'triple': triple_enabled
        }
    )

    # 백테스트 실행 버튼
    if st.sidebar.button("백테스트 실행"):
        # 백테스트 실행
        with st.spinner("백테스트 실행 중..."):
            result = analyzer.run_backtest()
            
        if result is not None:
            # 성과 요약 표시
            st.header("📊 백테스트 성과 요약")
            summary = analyzer.get_summary_data()
            
            if summary:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="총 수익률", 
                        value=f"{summary['multi_total_return']:.2f}%",
                        delta=f"{summary['return_diff']:.2f}% vs B&H"
                    )
                    st.metric(
                        label="CAGR (연평균 수익률)", 
                        value=f"{summary['multi_cagr']:.2f}%",
                        delta=f"{summary['multi_cagr'] - summary['bh_cagr']:.2f}% vs B&H"
                    )
                    
                with col2:
                    st.metric(
                        label="샤프 지수", 
                        value=f"{summary['multi_sharpe']:.2f}",
                        delta=f"{summary['sharpe_diff']:.2f} vs B&H"
                    )
                    st.metric(
                        label="최대 손실폭 (MDD)", 
                        value=f"{summary['multi_max_dd']:.2f}%",
                        delta=f"{summary['max_dd_diff']:.2f}% vs B&H",
                        delta_color="inverse"
                    )
                    
                with col3:
                    st.metric(
                        label="Buy & Hold 수익률", 
                        value=f"{summary['bh_total_return']:.2f}%"
                    )
                    st.metric(
                        label="Buy & Hold CAGR", 
                        value=f"{summary['bh_cagr']:.2f}%"
                    )
            
                # 누적 수익률 차트
                st.header("📈 누적 수익률 차트")
                cumulative_fig = analyzer.create_cumulative_returns_chart()
                if cumulative_fig:
                    st.plotly_chart(cumulative_fig, use_container_width=True)
                    
                # 월별 수익률 분석
                st.header("📅 월별 수익률 분석")
                monthly_data = analyzer.get_monthly_returns_data()
                
                if monthly_data and "error" not in monthly_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="월평균 예상 수익률", 
                            value=f"{monthly_data['expected_monthly_return_pct']:.2f}%"
                        )
                        st.metric(
                            label="월 평균 수익 (원)", 
                            value=f"{monthly_data['expected_monthly_return_krw']:,} 원"
                        )
                        
                    with col2:
                        st.metric(
                            label="승률", 
                            value=f"{monthly_data['winning_months_pct']:.1f}%",
                            delta=f"{monthly_data['winning_months_pct'] - monthly_data['bh_winning_months_pct']:.1f}% vs B&H"
                        )
                        st.metric(
                            label="Buy & Hold 대비 우위", 
                            value=f"{monthly_data['better_than_bh_pct']:.1f}%"
                        )
                        
                    # 월별 히트맵
                    st.subheader("월별 수익률 히트맵")
                    heatmap_fig = analyzer.create_monthly_heatmap()
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                elif monthly_data and "error" in monthly_data:
                    st.warning(monthly_data["error"])
                    
                # 종합 분석 차트
                st.header("📊 종합 분석 차트")
                detailed_fig = analyzer.create_detailed_analysis_chart()
                if detailed_fig:
                    st.pyplot(detailed_fig)
            else:
                st.error("성과 데이터를 계산할 수 없습니다.")
        else:
            st.error("백테스트 실행 중 오류가 발생했습니다. 설정을 확인하고 다시 시도해주세요.")
    else:
        # 앱이 처음 실행될 때 기본 안내 메시지 표시
        st.info("👈 왼쪽 사이드바에서 백테스트 파라미터를 설정하고 '백테스트 실행' 버튼을 클릭하세요.")